from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import datetime
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

os.makedirs("incidents", exist_ok=True)
os.makedirs("templates", exist_ok=True)

threat_data = {"score": 0, "status": "ALL CLEAR", "total_incidents": 0}

def get_threat_score(detections):
    score = 0
    for label in detections:
        if label == "person": score += 3
        if label == "knife": score += 8
        if label == "scissors": score += 5
        if label == "cell phone": score += 1
    return min(score, 10)

def generate_frames():
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

                if get_threat_score([label]) >= 5:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.0%}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        score = get_threat_score(detected_labels)
        threat_data["score"] = score

        if score >= 7:
            threat_data["status"] = "HIGH THREAT"
            threat_data["total_incidents"] += 1
        elif score >= 4:
            threat_data["status"] = "MEDIUM THREAT"
        else:
            threat_data["status"] = "ALL CLEAR"

        color = (0, 0, 255) if score >= 7 else (0, 165, 255) if score >= 4 else (0, 255, 0)
        cv2.putText(frame, f"THREAT: {score}/10",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(threat_data)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
