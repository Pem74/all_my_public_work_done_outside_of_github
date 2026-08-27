import cv2
from ultralytics import YOLO

def detect_people(video_path, output_path='output0.mp4'):
    # Charger le modèle YOLOv8 pré-entraîné (nano)
    model = YOLO("yolov8n.pt")

    # Ouvrir le flux vidéo
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Faire l'inférence sur le frame
        results = model(frame)

        # Extraire les boîtes englobantes de la première image (puisqu'il s'agit d'une seule image dans ce cas)
        for result in results:  # results est une liste de `Result` objets
            for box in result.boxes:  # Accéder aux boîtes englobantes
                xyxy = box.xyxy[0].cpu().numpy()  # Coordonnées de la boîte
                conf = box.conf.cpu().numpy()  # Confiance de la prédiction
                cls = box.cls.cpu().numpy()  # Classe prédite (numérique)
                label = model.names[int(cls)]  # Nom de la classe

                # Si la classe prédite est une personne
                if label.lower() == 'person':
                    # Dessiner les boîtes englobantes
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 3)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Afficher et sauvegarder les résultats
        cv2.imshow("processed frame", frame)
        out.write(frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Spécifier le chemin vidéo ou l'URL du flux
    video_path = "http://192.168.41.75:5000/video_feed"  # Par exemple, un flux réseau
    detect_people(video_path)