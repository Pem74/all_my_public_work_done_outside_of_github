import cv2
import requests
import numpy as np

# URL du flux vidéo (le serveur Flask que vous avez mis en place)
url = "http://192.168.4.75:5000/video_feed"

# Envoyer une requête au serveur pour obtenir le flux
stream = requests.get(url, stream=True)

# Accumulateur pour les données binaires de l'image
byte_data = b''

# Boucle sur le flux vidéo
for chunk in stream.iter_content(chunk_size=4096):
    # Ajouter les données au flux actuel
    byte_data += chunk

    # Rechercher la fin de l'en-tête multipart
    start_frame = byte_data.find(b'\xff\xd8')  # Marqueur de début JPEG
    end_frame = byte_data.find(b'\xff\xd9')    # Marqueur de fin JPEG
    #print(start_frame, end_frame)

    if start_frame != -1 and end_frame != -1:
        # Extraire les données JPEG
        frame_data = byte_data[start_frame:end_frame+2]  # Inclure le marqueur de fin JPEG
        byte_data = byte_data[end_frame+2:]  # Supprimer l'image extraite du buffer

        # Convertir les bytes en une image OpenCV
        np_frame = np.frombuffer(frame_data, np.uint8)
        image = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

        # Si l'image a été correctement décodée
        if image is not None:
            # Afficher l'image
            cv2.imshow("Stream", image)

            # Exemple d'analyse : affichage des bords
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            cv2.imshow("Edges", edges)

            # Quitter la boucle si on appuie sur 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Libérer les ressources
cv2.destroyAllWindows()