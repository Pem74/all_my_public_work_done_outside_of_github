import cv2

def start_segmentation(segmenter):
    vc = cv2.VideoCapture(0)
    while True:
        success, frame = vc.read()
        if not success:
            print("oula")
            continue
        cv2.imshow('frame', segmenter(frame))
        cv2.waitKey(1)

if __name__ == '__main__':
    start_segmentation(lambda x: x)