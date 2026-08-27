import cv2
from flask import Flask, Response, jsonify
from ultralytics import YOLO
import threading

app = Flask(__name__)

# Modèle YOLOv8
model = YOLO('yolov8n.pt')


cap = cv2.VideoCapture('http://192.168.4.75:5000/video_feed')
# Vérifier que le flux vidéo est bien ouvert
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir le flux vidéo.")

# Définir class_counts comme une variable globale pour stocker le nombre d'objets détectés
class_counts = {}


@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global class_counts  # Déclaration globale pour utiliser et modifier la variable

        while True:
            # Lire le flux vidéo
            success, frame = cap.read()

            if not success:
                print("Erreur : impossible de lire la vidéo.")
                break

            # Traiter le frame avec YOLOv8 (à adapter selon ton modèle YOLO)
            results = model(frame)  # Assure-toi que ton modèle est chargé avant

            # Réinitialiser le compteur des classes pour chaque nouvelle frame
            class_counts = {}

            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf.cpu().numpy()
                    cls = box.cls.cpu().numpy()
                    label = model.names[int(cls)]

                    # Compter les classes détectées
                    if label in class_counts:
                        class_counts[label] += 1
                    else:
                        class_counts[label] = 1

                    # Dessiner les boîtes de détection sur l'image
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {float(conf):.2f}', (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Encoder l'image en JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Erreur : échec de l'encodage du frame.")
                break

            frame = buffer.tobytes()

            # Envoyer le frame sous forme de flux vidéo MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection_results')
def detection_results():
    global class_counts  # Utiliser la variable globale class_counts
    return jsonify(class_counts)


if __name__ == '__main__':
    # Démarrer l'application Flask
    app.run(host='0.0.0.0', port=8080)