import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

print("CRIMEVIS AI - Person Detection Active...")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame, verbose=False)
    
    for result in results:
        for box in result.boxes:
            # Only detect person (class 0)
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf)
                
                # Draw green box around person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"PERSON {confidence:.0%}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, "CRIMEVIS AI - PERSON DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("CRIMEVIS AI - Person Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
