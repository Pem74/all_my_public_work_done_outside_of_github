import streamlit as st
from PIL import Image
import io
import requests

#  streamlit run C:/Cours/SN/snA24/IA03/Projet/flask_streamlit/frontend.py

st.title("Poubelle Intelligente")

upload = st.file_uploader("Chargez l'image de votre objet",
                           type=['png', 'jpeg', 'jpg'])

c1, c2 = st.columns(2)

if upload:
    files = {"file": upload.getvalue()}

    # Faire la demande POST à l'API
    response = requests.post("http://127.0.0.1:8080/predict", files=files)

    # Vérifier que la réponse est valide
    if response.status_code == 200:
        # Lire l'image depuis les octets
        img = Image.open(io.BytesIO(response.content))

        # Afficher l'image traitée
        c1.image(img, caption="Image Traitée", use_column_width=True)
    else:
        st.error("Erreur lors du traitement de l'image")
