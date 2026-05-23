import cv2
from ultralytics import YOLO
import time
import datetime

model = YOLO("yolov8n.pt")
model.overrides['conf'] = 0.5

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("CRIMEVIS AI - Fall Detection Active...")

fall_start_time = None
FALL_THRESHOLD = 3

def is_fallen(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    # Person is fallen if width > height
    # Standing: height > width
    # Fallen: width > height (horizontal)
    ratio = width / height if height > 0 else 0
    return ratio > 1.2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    fall_detected = False
    emergency = False

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if is_fallen((x1, y1, x2, y2)):
                    fall_detected = True
                    cv2.rectangle(frame,
                                  (x1, y1), (x2, y2),
                                  (0, 0, 255), 3)
                    cv2.putText(frame, "FALLEN PERSON",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame,
                                  (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, "Person - Standing",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

    if fall_detected:
        if fall_start_time is None:
            fall_start_time = time.time()
        else:
            time_fallen = time.time() - fall_start_time
            if time_fallen >= FALL_THRESHOLD:
                emergency = True
    else:
        fall_start_time = None

    if emergency:
        cv2.putText(frame, "MEDICAL EMERGENCY - PERSON DOWN!",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.putText(frame, "ALERTING AMBULANCE...",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        print("MEDICAL EMERGENCY DETECTED!")

    elif fall_detected:
        cv2.putText(frame, "WARNING: FALL DETECTED",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 165, 255), 2)

    else:
        cv2.putText(frame, "MONITORING - ALL CLEAR",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("CRIMEVIS AI - Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
