import streamlit as st
import requests
from streamlit.components.v1 import html

#  streamlit run C:/Cours/SN/snA24/IA03/Projet/flask_streamlit/frontend_flux.py

st.title("Flux vidéo traité avec YOLOv8")

# Afficher le flux vidéo provenant du backend Flask
video_url = "http://127.0.0.1:8080/video_feed"

html(f"""
    <div style="display: flex; justify-content: center;">
        <img src="{video_url}" width="640" height="480" alt="Flux vidéo traité">
    </div>
""", height=500)

# Afficher les résultats de détection en temps réel
if st.button("Obtenir les résultats de détection"):
    response = requests.get("http://127.0.0.1:8080/detection_results")
    if response.status_code == 200:
        detection_results = response.json()
        st.write("Classes détectées :")
        st.write(detection_results)
    else:
        st.write("Erreur lors de la récupération des résultats")