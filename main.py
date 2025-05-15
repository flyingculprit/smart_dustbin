import torch
import cv2
import pygame

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.eval()

# Initialize video capture
cap = cv2.VideoCapture(0)  # Webcam

# Initialize alarm
pygame.mixer.init()
pygame.mixer.music.load("alert.mp3")  # Your alarm sound file

alarm_playing = False

# Region of Interest (adjust based on dustbin location)
roi_x, roi_y, roi_w, roi_h = 300, 250, 200, 200

# Only trigger alarm for these classes (living beings)
allowed_classes = ['person', 'cat', 'dog']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)
    alarm_triggered = False

    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(cls)]

        # Skip if class is not a living being
        if class_name not in allowed_classes:
            continue

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Check if center point is inside dustbin (ROI)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if roi_x < cx < roi_x + roi_w and roi_y < cy < roi_y + roi_h:
            # Inside bin
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Inside Bin!", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            alarm_triggered = True

    # Play or stop alarm
    if alarm_triggered:
        if not alarm_playing:
            pygame.mixer.music.play(-1)  # -1 means loop until stopped
            alarm_playing = True
    else:
        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    # Show the frame
    cv2.imshow("Dustbin Monitor", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
