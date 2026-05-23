import cv2
from ultralytics import YOLO
import datetime
import time
import os

print("CRIMEVIS AI - Gesture SOS Detection Active...")

model = YOLO("yolov8n.pt")
model.overrides['conf'] = 0.5

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

os.makedirs("incidents", exist_ok=True)

# SOS tracking
sos_start_time = None
SOS_THRESHOLD = 2  # 2 seconds of raised hands = SOS
last_alert = None

# Track person top position history
person_top_history = []

def detect_hands_raised(box_history):
    """
    If person's bounding box TOP keeps moving UP
    = hands being raised above head
    """
    if len(box_history) < 8:
        return False

    recent = box_history[-8:]
    # Get y1 positions (top of person box)
    tops = [b[1] for b in recent]
    # If top of box is consistently high up in frame
    # and box is tall (standing person with hands up)
    avg_top = sum(tops) / len(tops)

    # Person top is in upper 40% of frame = hands raised
    return avg_top < 150

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    now = datetime.datetime.now()

    hands_raised = False

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Track box history
                person_top_history.append((x1, y1, x2, y2))
                if len(person_top_history) > 20:
                    person_top_history.pop(0)

                hands_raised = detect_hands_raised(
                    person_top_history)

                color = (0, 0, 255) if hands_raised else (0, 255, 0)
                cv2.rectangle(frame,
                              (x1, y1), (x2, y2), color, 2)

                if hands_raised:
                    cv2.putText(frame, "HANDS RAISED!",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

    # SOS Logic
    SOS_TRIGGERED = False
    if hands_raised:
        if sos_start_time is None:
            sos_start_time = time.time()
        else:
            time_raised = time.time() - sos_start_time
            if time_raised >= SOS_THRESHOLD:
                SOS_TRIGGERED = True
    else:
        sos_start_time = None

    # Display
    if SOS_TRIGGERED:
        cv2.putText(frame,
                    "SOS! PERSON NEEDS HELP!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
        cv2.putText(frame,
                    "ALERTING SECURITY NOW...",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

        if last_alert is None or (now - last_alert).seconds >= 30:
            filename = f"incidents/sos_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"SOS ALERT SAVED: {filename}")
            last_alert = now

    elif hands_raised:
        cv2.putText(frame,
                    "WARNING: HANDS RAISED - MONITORING",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2)

    else:
        cv2.putText(frame,
                    "MONITORING - ALL CLEAR",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp,
                (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

    cv2.imshow("CRIMEVIS AI - Gesture SOS", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
