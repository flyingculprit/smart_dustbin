import cv2
import torch
import pygame

# Initialize Pygame mixer for alarm sound
pygame.mixer.init()
pygame.mixer.music.load("alert.mp3")  # Put your alarm sound file here
alarm_playing = False

# Load YOLOv5 model (use yolov5s, or your trained custom model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define ROI for dustbin (adjust as needed)
roi_x, roi_y, roi_w, roi_h = 300, 150, 300, 300

# Allowed classes for living beings to trigger alarm
allowed_classes = ['cat', 'dog', 'person', 'child']  # 'child' might not exist, 'person' covers humans

# Classes to consider as trash for fullness calculation (add more if your model detects)
trash_classes = ['bottle', 'cup', 'box', 'plastic', 'bag', 'cell phone', 'book', 'banana']  # example trash classes

# Threshold for considering dustbin full (70%)
FILL_THRESHOLD = 0.7

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference with YOLOv5
    results = model(frame)

    # Draw dustbin ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)

    alarm_triggered = False
    occupied_area = 0
    roi_area = roi_w * roi_h

    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(cls)]

        # Check for living beings inside dustbin ROI for alarm
        if class_name in allowed_classes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if roi_x < cx < roi_x + roi_w and roi_y < cy < roi_y + roi_h:
                alarm_triggered = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Inside Bin!", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Check for trash objects inside dustbin ROI for fullness
        if class_name in trash_classes:
            # Clamp bounding box to dustbin ROI
            box_x1 = max(x1, roi_x)
            box_y1 = max(y1, roi_y)
            box_x2 = min(x2, roi_x + roi_w)
            box_y2 = min(y2, roi_y + roi_h)

            if box_x2 > box_x1 and box_y2 > box_y1:
                box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
                occupied_area += box_area
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 165, 255), 2)
                cv2.putText(frame, class_name, (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Calculate fill ratio
    fill_ratio = occupied_area / roi_area

    # Display fill status on frame
    fill_text = f"Dustbin Fill: {fill_ratio*100:.1f}%"
    if fill_ratio > FILL_THRESHOLD:
        fill_text += " - FULL"
        cv2.putText(frame, fill_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, fill_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Play or stop alarm sound
    if alarm_triggered:
        if not alarm_playing:
            pygame.mixer.music.play(-1)  # Loop alarm sound
            alarm_playing = True
    else:
        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    cv2.imshow("Dustbin Monitor", frame)

    if cv2.waitKey(1) == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()
