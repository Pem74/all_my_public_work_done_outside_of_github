import cv2
from flask import Flask, Response
import torch
from ultralytics import YOLO

app = Flask(__name__)

def generate_det_frames():
    model = YOLO('yolov8n-pose.pt')

    # Palette de couleurs pour distinguer les différentes personnes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
    ]

    # Connexions entre les points clés (par exemple, pour COCO keypoints)
    connections = [
        (0, 1), (1, 3), (0, 2), (2, 4),  # Tête et épaules
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Bras
        (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Jambes
    ]

    # Ouvrir la vidéo
    cap = cv2.VideoCapture('http://localhost:8080/video_feed')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))

        # Prédire les poses dans le cadre
        results = model(frame)
        keypoints = results[0].keypoints.xy.cpu().numpy()
        confs = results[0].keypoints.conf.cpu().numpy()

        for i, kp in enumerate(keypoints):
            color = colors[i % len(colors)]  # Utiliser une couleur différente pour chaque personne
            for j, (x, y) in enumerate(kp):
                if confs[i, j] > 0.5:  # Seuil de confiance pour dessiner les points clés
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)

            # Dessiner les connexions entre les points clés
            for (start, end) in connections:
                if confs[i, start] > 0.5 and confs[i, end] > 0.5:
                    cv2.line(frame, (int(kp[start][0]), int(kp[start][1])),
                             (int(kp[end][0]), int(kp[end][1])), color, 2)
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the output frame in byte format
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_detect')
def video_detect():
    # Route for the video with detections
    return Response(generate_det_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')
# def index():
#     # Main page with the video feed
#     return '''
#     <html>
#     <head>
#         <title>USB Camera Stream</title>
#     </head>
#     <body>
#         <h1>USB Camera Stream</h1>
#         <img src="/video_feed" width="640" height="480">
#     </body>
#     </html>
#     '''

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    # Démarrer le serveur Flask sur localhost et le port 5000
    app.run(host='0.0.0.0', port=5000)