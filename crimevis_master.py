from email_alert import send_alert
import cv2
from ultralytics import YOLO
import datetime
import os
import time
import threading
import pyaudio
import numpy as np

# ── Setup ──────────────────────────────
print("CRIMEVIS AI - Master System Loading...")

model = YOLO("yolov8n.pt")
model.overrides['conf'] = 0.5

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml')

os.makedirs("incidents", exist_ok=True)

# ── Settings ───────────────────────────
LOITERING_THRESHOLD  = 60
MOVEMENT_THRESHOLD   = 50
ALERT_COOLDOWN       = 60
THREAT_ALERT_LEVEL   = 7
FALL_THRESHOLD       = 3
SOS_THRESHOLD        = 2
NO_FACE_THRESHOLD    = 15
SCREAM_THRESHOLD     = 3000

# ── Global Flags ───────────────────────
audio_alert      = False
scream_volume    = 0

# ── Audio Thread ───────────────────────
def listen_audio():
    global audio_alert, scream_volume
    try:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024)
        print("Audio monitoring active...")
        while True:
            try:
                data = stream.read(
                    1024, exception_on_overflow=False)
                audio_data = np.frombuffer(
                    data, dtype=np.int16)
                volume = np.abs(audio_data).mean()
                scream_volume = volume
                audio_alert = volume > SCREAM_THRESHOLD
                if audio_alert:
                    print(f"SCREAM! Volume:{volume:.0f}")
            except:
                pass
    except Exception as e:
        print(f"Audio error: {e}")

audio_thread = threading.Thread(
    target=listen_audio, daemon=True)
audio_thread.start()

# ── Trackers ───────────────────────────
person_first_seen  = None
prev_boxes         = []
last_saved         = None
fall_start_time    = None
sos_start_time     = None
no_face_start      = None
collapse_start     = None
person_top_history = []

# ── Threat Score ───────────────────────
def get_threat_score(labels, loitering=False,
    fight=False, fall=False, sos=False,
    no_face=False, scream=False, collapse=False):
    score = 0
    for l in labels:
        if l == "person":     score += 3
        if l == "knife":      score += 8
        if l == "scissors":   score += 5
        if l == "cell phone": score += 1
    if loitering: score += 3
    if fight:     score += 6
    if fall:      score += 7
    if sos:       score += 8
    if no_face:   score += 5
    if scream:    score += 7
    if collapse:  score += 9
    return min(score, 10)

def get_color(score):
    if score <= 3:   return (0, 255, 0)
    elif score <= 6: return (0, 165, 255)
    else:            return (0, 0, 255)

# ── Fall Detection ─────────────────────
def is_fallen(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return (w/h if h > 0 else 0) > 1.2

# ── Fight Detection ────────────────────
def detect_fight(curr, prev):
    if not curr or not prev:
        return False
    movement = sum(
        abs(curr[i][0]-prev[i][0]) +
        abs(curr[i][1]-prev[i][1])
        for i in range(min(len(curr), len(prev))))
    return movement > MOVEMENT_THRESHOLD

# ── SOS Detection ──────────────────────
def detect_sos(history):
    if len(history) < 8:
        return False
    tops = [b[1] for b in history[-8:]]
    return (sum(tops)/len(tops)) < 150

print("All systems ready! Starting camera...")

# ── Main Loop ──────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = datetime.datetime.now()
    results = model(frame, verbose=False)

    detected_labels = []
    current_boxes   = []
    person_detected = False
    fall_detected   = False

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            conf  = float(box.conf)
            detected_labels.append(label)
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            current_boxes.append((x1,y1,x2,y2))

            if label == "person":
                person_detected = True
                person_top_history.append(
                    (x1,y1,x2,y2))
                if len(person_top_history) > 20:
                    person_top_history.pop(0)

                if is_fallen((x1,y1,x2,y2)):
                    fall_detected = True
                    cv2.rectangle(frame,
                        (x1,y1),(x2,y2),
                        (0,0,255),3)
                    cv2.putText(frame,"FALLEN!",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,0,255),2)
                else:
                    cv2.rectangle(frame,
                        (x1,y1),(x2,y2),
                        (0,255,0),2)
                    cv2.putText(frame,
                        f"Person {conf:.0%}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)
            else:
                cv2.rectangle(frame,
                    (x1,y1),(x2,y2),
                    (0,165,255),2)
                cv2.putText(frame,
                    f"{label} {conf:.0%}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(0,165,255),2)

    # ── Loitering ──────────────────────
    loitering = False
    if person_detected:
        if person_first_seen is None:
            person_first_seen = time.time()
        elif time.time()-person_first_seen > \
                LOITERING_THRESHOLD:
            loitering = True
    else:
        person_first_seen = None

    # ── Fight ───────────────────────────
    fight = detect_fight(current_boxes, prev_boxes)
    prev_boxes = current_boxes

    # ── Fall Emergency ──────────────────
    fall_emergency = False
    if fall_detected:
        if fall_start_time is None:
            fall_start_time = time.time()
        elif time.time()-fall_start_time > \
                FALL_THRESHOLD:
            fall_emergency = True
    else:
        fall_start_time = None

    # ── SOS ─────────────────────────────
    sos = False
    if detect_sos(person_top_history):
        if sos_start_time is None:
            sos_start_time = time.time()
        elif time.time()-sos_start_time > \
                SOS_THRESHOLD:
            sos = True
    else:
        sos_start_time = None

    # ── Face / Unconscious ──────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 1.1, 5, minSize=(30,30))
    face_detected = len(faces) > 0

    no_face_alert = False
    if not face_detected and person_detected:
        if no_face_start is None:
            no_face_start = time.time()
        elif time.time()-no_face_start > \
                NO_FACE_THRESHOLD:
            no_face_alert = True
    else:
        no_face_start = None

    # ── Collapse Detection ──────────────
    # Person fallen + no face = collapsed/unconscious
    collapse = False
    if fall_detected and not face_detected:
        if collapse_start is None:
            collapse_start = time.time()
        elif time.time()-collapse_start > 3:
            collapse = True
            print("COLLAPSE/UNCONSCIOUS DETECTED!")
    else:
        collapse_start = None

    # ── Threat Score ────────────────────
    score = get_threat_score(
        detected_labels,
        loitering, fight,
        fall_emergency, sos,
        no_face_alert, audio_alert,
        collapse)
    color = get_color(score)

    # ── Display ─────────────────────────
    y = 30
    cv2.putText(frame,
        f"CRIMEVIS AI | THREAT: {score}/10",
        (10,y), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, color, 2)
    y += 35

    if collapse:
        cv2.putText(frame,
            "COLLAPSE! PERSON UNCONSCIOUS!",
            (10,y), cv2.FONT_HERSHEY_SIMPLEX,
            0.8,(0,0,255),2)
        y += 30

    if fall_emergency:
        cv2.putText(frame,
            "MEDICAL EMERGENCY - PERSON DOWN!",
            (10,y), cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0,0,255),2)
        y += 30

    if sos:
        cv2.putText(frame,
            "SOS! PERSON NEEDS HELP!",
            (10,y), cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0,0,255),2)
        y += 30

    if audio_alert:
        cv2.putText(frame,
            f"SCREAM DETECTED! Vol:{scream_volume:.0f}",
            (10,y), cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0,0,255),2)
        y += 30

    if fight:
        cv2.putText(frame,
            "FIGHT/AGGRESSION DETECTED!",
            (10,y), cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0,0,255),2)
        y += 30

    if loitering:
        cv2.putText(frame,
            "WARNING: LOITERING DETECTED",
            (10,y), cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0,165,255),2)
        y += 30

    if no_face_alert:
        cv2.putText(frame,
            "UNCONSCIOUS PERSON ALERT!",
            (10,y), cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0,0,255),2)
        y += 30

    if score < 4:
        cv2.putText(frame, "ALL CLEAR",
            (10,y), cv2.FONT_HERSHEY_SIMPLEX,
            0.8,(0,255,0),2)

    # Volume bar
    bar = int(scream_volume/100)
    cv2.putText(frame,
        f"Audio: {'|'*min(bar,20)}",
        (10, frame.shape[0]-30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,(255,255,0),1)

    # Timestamp
    cv2.putText(frame,
        now.strftime("%Y-%m-%d %H:%M:%S"),
        (10, frame.shape[0]-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,(255,255,255),1)

    # ── Save & Alert ────────────────────
    if score >= THREAT_ALERT_LEVEL:
        if last_saved is None or \
                (now-last_saved).seconds >= ALERT_COOLDOWN:
            fname = f"incidents/master_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, frame)
            print(f"INCIDENT SAVED: {fname}")
            send_alert(score, fname)
            last_saved = now

    cv2.imshow("CRIMEVIS AI - Master System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("CRIMEVIS AI stopped.")
