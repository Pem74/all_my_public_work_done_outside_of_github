import cv2
import numpy as np
from torch.hub import load as hub_load

# Define the vertex coordinates of the polygon region
pts = np.array([[0, 0],  [220, 0], [60, 400], [30, 480],[0, 480]  ], np.int32)
pts = pts.reshape((-1, 1, 2))

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Get the width and height of the webcam frames
width = int(cap.get(3))
height = int(cap.get(4))

# Initialize the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (width, height))

# Initialize the YOLOv5 model
model = hub_load('ultralytics/yolov5:master', 'yolov5s', pretrained=True)
model.eval()

while True:
    ret, frame = cap.read()

    # Use the YOLOv5 model to detect people
    results = model(frame)
    pred = results.xyxy[0].cpu().numpy()

    for xyxy in pred:
        label = model.names[int(xyxy[5])]

        # Process only targets with the 'person' label
        if label == 'person':
            x, y, w, h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # Check if the bounding box is inside the polygon; if yes, change the box color to red
            if cv2.pointPolygonTest(pts, (x + w // 2, y + h // 2), False) >= 0:
                cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw the polygon on the video frame
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

    # Write the frame to the video file
    out.write(frame)

    # Display the result
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoWriter and close the OpenCV window
out.release()
cap.release()
cv2.destroyAllWindows()
