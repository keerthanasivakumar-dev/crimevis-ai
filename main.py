from email_alert import send_alert
import cv2
from ultralytics import YOLO
import datetime
import os
import time

model = YOLO("yolov8n.pt")
model.overrides['conf'] = 0.5

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

os.makedirs("incidents", exist_ok=True)

print("CRIMEVIS AI - Full Detection Active...")

# Settings
LOITERING_THRESHOLD = 60  # 60 seconds
MOVEMENT_THRESHOLD = 50
ALERT_COOLDOWN = 60  # 60 seconds between alerts
THREAT_ALERT_LEVEL = 7  # only alert score 7+

person_first_seen = None
prev_boxes = []
last_saved = None

def get_threat_score(detections, loitering=False, fight=False):
    score = 0
    for label in detections:
        if label == "person":     score += 3
        if label == "knife":      score += 8
        if label == "scissors":   score += 5
        if label == "cell phone": score += 1
    if loitering: score += 4
    if fight:     score += 6
    return min(score, 10)

def get_threat_color(score):
    if score <= 3:   return (0, 255, 0)
    elif score <= 6: return (0, 165, 255)
    else:            return (0, 0, 255)

def detect_fight(current_boxes, previous_boxes):
    if len(current_boxes) == 0 or len(previous_boxes) == 0:
        return False
    total_movement = 0
    for i in range(min(len(current_boxes), len(previous_boxes))):
        cx = abs(current_boxes[i][0] - previous_boxes[i][0])
        cy = abs(current_boxes[i][1] - previous_boxes[i][1])
        total_movement += cx + cy
    return total_movement > MOVEMENT_THRESHOLD

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detected_labels = []
    current_boxes = []
    person_detected = False

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            confidence = float(box.conf)
            detected_labels.append(label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_boxes.append((x1, y1, x2, y2))
            if label == "person":
                person_detected = True
            color = get_threat_color(3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.0%}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Loitering Detection
    loitering = False
    if person_detected:
        if person_first_seen is None:
            person_first_seen = time.time()
        else:
            time_in_frame = time.time() - person_first_seen
            if time_in_frame > LOITERING_THRESHOLD:
                loitering = True
    else:
        person_first_seen = None

    # Fight Detection
    fight = detect_fight(current_boxes, prev_boxes)
    prev_boxes = current_boxes

    # Threat Score
    score = get_threat_score(detected_labels, loitering, fight)
    color = get_threat_color(score)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Display
    cv2.putText(frame, f"THREAT SCORE: {score}/10",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, timestamp,
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    if loitering:
        cv2.putText(frame, "WARNING: LOITERING DETECTED",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2)

    if fight:
        cv2.putText(frame, "ALERT: FIGHT DETECTED",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    if score >= THREAT_ALERT_LEVEL:
        cv2.putText(frame, "!! HIGH THREAT - SAVING !!",
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        if last_saved is None or (now - last_saved).seconds >= ALERT_COOLDOWN:
            filename = f"incidents/incident_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"INCIDENT SAVED: {filename}")
            send_alert(score, filename)
            last_saved = now
    else:
        cv2.putText(frame, "ALL CLEAR",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("CRIMEVIS AI - Full Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

