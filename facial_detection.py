import cv2
import datetime
import os
import time

print("CRIMEVIS AI - Facial Detection Active...")

os.makedirs("incidents", exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load face and eye detectors
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

last_alert = None
no_face_start = None
NO_FACE_THRESHOLD = 10  # seconds without face = unconscious alert
face_history = []

print("Monitoring faces...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = datetime.datetime.now()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))

    face_detected = len(faces) > 0
    eyes_open = False
    distress = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Check eyes inside face region
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        eyes_open = len(eyes) >= 2

        if eyes_open:
            cv2.putText(frame, "EYES OPEN",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "EYES CLOSED - MONITORING",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 165, 255), 2)
            distress = True

        # Draw eyes
        roi_color = frame[y:y+h, x:x+w]
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,
                          (ex, ey), (ex+ew, ey+eh),
                          (0, 255, 255), 2)

    # Track face history
    face_history.append(face_detected)
    if len(face_history) > 100:
        face_history.pop(0)

    # No face for long time = unconscious
    if not face_detected:
        if no_face_start is None:
            no_face_start = time.time()
        else:
            no_face_time = time.time() - no_face_start
            if no_face_time > NO_FACE_THRESHOLD:
                cv2.putText(frame,
                            "ALERT: PERSON MAY BE UNCONSCIOUS!",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

                if last_alert is None or \
                        (now - last_alert).seconds >= 30:
                    filename = f"incidents/facial_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"UNCONSCIOUS ALERT SAVED: {filename}")
                    last_alert = now
    else:
        no_face_start = None

    # Display status
    if face_detected and distress:
        cv2.putText(frame, "WARNING: EYES CLOSED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 165, 255), 2)
    elif face_detected:
        cv2.putText(frame, "MONITORING - FACE DETECTED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO FACE DETECTED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"),
                (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

    cv2.imshow("CRIMEVIS AI - Facial Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


