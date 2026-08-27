import torch
import cv2
import matplotlib.backends
from torch.hub import load as hub_load



def detect_people(video_path, output_path='output0.mp4'):
    model = hub_load('ultralytics/yolov5:master', 'yolov5s', pretrained=True)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        pred = results.xyxy[0].cpu().numpy()

        for xyxy in pred:
            #label = f'{model.names[int(xyxy[5])]} {xyxy[4]:.2f}'  # affiche proba, mettre du coup la ligne suivante en commentaire
            label = model.names[int(xyxy[5])]
            if label.lower() == 'person':
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 3)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #out.write(frame)
        cv2.imshow("processed frame",frame)
        cv2.waitKey(1)

    cap.release()
    #out.release()

if __name__ == '__main__':
    #video_path = 0 # 'C:\\Users\\fengyizh\\Videos\\Logitech Webcam\\Video 1.mp4'
    video_path = 'Video_Data_Days.mp4'
    detect_people(video_path)
    #detect_people(1)
    
