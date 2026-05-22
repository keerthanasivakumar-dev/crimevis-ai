import cv2
from ultralytics import YOLO
import datetime
import os

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Create folder to save incidents
os.makedirs("incidents", exist_ok=True)

print("CRIMEVIS AI - Incident Logging Active...")

def get_threat_score(detections):
    score = 0
    for label in detections:
        if label == "person": score += 3
        if label == "knife": score += 8
        if label == "scissors": score += 5
        if label == "cell phone": score += 1
    return min(score, 10)

def get_threat_color(score):
    if score <= 3: return (0, 255, 0)
    elif score <= 6: return (0, 165, 255)
    else: return (0, 0, 255)

last_saved = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detected_labels = []

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            confidence = float(box.conf)
            detected_labels.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = get_threat_color(3)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.0%}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    score = get_threat_score(detected_labels)
    color = get_threat_color(score)

    # Timestamp
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    cv2.putText(frame, f"THREAT SCORE: {score}/10",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, timestamp,
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    if score >= 4:
        cv2.putText(frame, "!! THREAT DETECTED - SAVING !!",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

        # Save screenshot every 5 seconds
        if last_saved is None or (now - last_saved).seconds >= 5:
            filename = f"incidents/incident_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"INCIDENT SAVED: {filename}")
            last_saved = now
    else:
        cv2.putText(frame, "ALL CLEAR",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("CRIMEVIS AI - Incident Logging", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
