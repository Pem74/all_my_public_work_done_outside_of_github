import cv2
import random
from ultralytics import YOLO
from flask import Flask, jsonify
from threading import Thread

#  python -m flask --app YOLO_API run

app = Flask(__name__)

# Pose connections (comme avant)
POSE_CONNECTIONS = [
    (0, 1), (1, 3), (0, 2), (2, 4),  # Tête et épaules
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Bras
    (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Jambes
]

poses_data = []  # Liste globale pour stocker les données des poses


def assign_color(person_id, person_colors):
    if person_id not in person_colors:
        person_colors[person_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return person_colors[person_id]


def detect_poses(video_path, output_path='output0.mp4'):
    global poses_data  # Accéder aux données globales des poses
    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    person_colors = {}  # Pour stocker les couleurs par ID de personne
    previous_boxes = []  # Pour stocker les boîtes englobantes de la frame précédente

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        current_boxes = []
        poses_data = []  # Réinitialiser les données des poses pour chaque frame

        for result in results:
            if len(result.keypoints) == 0:
                continue

            for box, kp in zip(result.boxes, result.keypoints):
                box_coords = tuple(box.xyxy[0].cpu().numpy())  # Convertir en tuple pour comparaison
                current_boxes.append(box_coords)

                # Vérifier si la boîte actuelle est proche de l'une des boîtes de la frame précédente
                found = False
                for prev_id, prev_box in enumerate(previous_boxes):
                    if iou(box_coords, prev_box) > 0.5:
                        person_id = prev_id
                        found = True
                        break

                if not found:
                    person_id = len(person_colors)

                color = assign_color(person_id, person_colors)

                # Récupérer les points clés
                person_keypoints = kp.data[0].cpu().numpy().tolist()

                # Stocker les poses pour l'API
                pose = {
                    'person_id': person_id,
                    'keypoints': person_keypoints
                }
                poses_data.append(pose)

                # Tracer les points clés et les connexions
                for (part_a, part_b) in POSE_CONNECTIONS:
                    if person_keypoints[part_a][2] > 0.5 and person_keypoints[part_b][2] > 0.5:
                        x1, y1 = int(person_keypoints[part_a][0]), int(person_keypoints[part_a][1])
                        x2, y2 = int(person_keypoints[part_b][0]), int(person_keypoints[part_b][1])
                        cv2.line(frame, (x1, y1), (x2, y2), color, 3)

                # Dessiner un cercle sur chaque point clé visible
                for keypoint in person_keypoints:
                    if keypoint[2] > 0.5:  # Confiance supérieure à 0.5
                        x, y = int(keypoint[0]), int(keypoint[1])
                        cv2.circle(frame, (x, y), 5, color, -1)

        # Mettre à jour les boîtes englobantes précédentes
        previous_boxes = current_boxes

        # Afficher et enregistrer le résultat
        cv2.imshow("processed frame", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def iou(box1, box2):
    """ Calcul de l'Intersection over Union (IoU) pour comparer les boîtes englobantes """
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)

    iou_value = inter_area / (box1_area + box2_area - inter_area)
    return iou_value


@app.route('/get_poses', methods=['GET'])
def get_poses():
    global poses_data
    return jsonify(poses_data)


class ProcessingThread(Thread):
    def run(self):
        video_path = "http://192.168.4.75:5000/video_feed"

        # Démarrer la détection des poses dans un thread séparé
        from threading import Thread

        thread = Thread(target=detect_poses, args=(video_path,))
        thread.start()

        # Démarrer l'API Flask
        app.run(host="0.0.0.0", port=5000)

ProcessingThread().start()