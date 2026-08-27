from flask import Flask, jsonify, request, Response
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2

app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello():
    return jsonify({"hello": "kevin"})

@app.before_first_request
def load():
    #model_path = "best_model.h5"
    #model = load_model(model_path, compile=False)
    model = YOLO('yolov8n.pt')
    return model

# Chargement du model
model = load()

def preprocess(img):
    img = img.resize((224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/predict", methods=['POST'])
def predict():
    # récupérer l'image
    file = request.files['file']
    image = file.read()

    # Ouvrir l'image
    img = Image.open(io.BytesIO(image))

    # Convertir l'image en format OpenCV pour le traitement
    img_cv = np.array(img)

    # Faire l'inférence sur le frame
    results = model(img_cv)

    # Dessiner les boîtes englobantes et les points clés
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf.cpu().numpy()
            cls = box.cls.cpu().numpy()
            label = model.names[int(cls)]

            if label.lower() == 'person':
                cv2.rectangle(img_cv, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 3)
                cv2.putText(img_cv, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

    # Convertir l'image traitée en JPEG
    _, img_encoded = cv2.imencode('.jpg', img_cv)
    img_bytes = img_encoded.tobytes()

    # Renvoyer l'image traitée
    return Response(img_bytes, mimetype='image/jpeg')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)