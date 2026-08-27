from flask import Flask, Response
import cv2
import requests
import numpy as np

app = Flask(__name__)

# Ouvrir la webcam
camera = cv2.VideoCapture(0)  # 0 pour la première webcam disponible

def generate_frames():

    while True:
        # Lire l'image de la webcam
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convertir le frame en flux binaire pour l'envoyer au client
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Créer une réponse de type image/jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Route pour diffuser le flux vidéo en continu
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Route principale, page web pour afficher la vidéo
    return '''
    <html>
        <head>
            <title>Flux Vidéo en Direct</title>
        </head>
        <body>
            <h1>Flux Vidéo en Direct</h1>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

