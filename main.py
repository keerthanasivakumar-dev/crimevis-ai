import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Draw a rectangle on screen
    cv2.rectangle(frame, (50, 50), (300, 300), (0, 255, 0), 2)
    
    # Add text on screen
    cv2.putText(frame, "CRIMEVIS AI ACTIVE", (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("CRIMEVIS AI - Live Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
