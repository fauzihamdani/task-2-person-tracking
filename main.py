import cv2
from ultralytics import YOLO
import numpy as np


cap = cv2.VideoCapture("person-tracking.mp4")

model = YOLO("yolov8s.pt")


className = ["person",
             "bicycle",
             "car",
             "motorcycle",
             "airplane",
             "bus",
             "train",
             "truck",
             "boat",
             "traffic light",
             "fire hydrant",
             "stop sign",
             "parking meter",
             "bench",
             "bird",
             "cat",
             "dog",
             "horse",
             "sheep",
             "cow",
             "elephant",
             "bear",
             "zebra",
             "giraffe",
             "backpack",
             "umbrella",
             "handbag",
             "tie",
             "suitcase",
             "frisbee",
             "skis",
             "snowboard",
             "sports ball",
             "kite",
             "baseball bat",
             "baseball glove",
             "skateboard",
             "surfboard",
             "tennis racket",
             "bottle",
             "wine glass",
             "cup",
             "fork",
             "knife",
             "spoon",
             "bowl",
             "banana",
             "apple",
             "sandwich",
             "orange",
             "broccoli",
             "carrot",
             "hot dog",
             "pizza",
             "donut",
             "cake",
             "chair",
             "couch",
             "potted plant",
             "bed",
             "dining table",
             "toilet",
             "tv",
             "laptop",
             "mouse",
             "remote",
             "keyboard",
             "cell phone",
             "microwave",
             "oven",
             "toaster",
             "sink",
             "refrigerator",
             "book",
             "clock",
             "vase",
             "scissors",
             "teddy bear",
             "hair drier",
             "toothbrush"]

while True:
    ret, frame = cap.read()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(frame, str(fps), (80, 80),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    if not ret:
        break

    results = model(frame, device='cpu')
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    class_name = model.predict(stream=True, imgsz=512)
    names = model.names

    for r in class_name:
        for c in r.boxes.cls:
            print(names[int(c)])
            # cv2.putText(frame, names, (x, y - 5),
            #             cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, str(
            className[cls]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
