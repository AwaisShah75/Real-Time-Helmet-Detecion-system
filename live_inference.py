import cv2
import math
import cvzone
from ultralytics import YOLO

# ---------------------------------------------------------
# Local Live Inference Script for Helmet Detection
# ---------------------------------------------------------

# Option 1: Load a local video file
video_path = "Media/traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Option 2: Open the webcam (Uncomment the line below to use your laptop camera)
# cap = cv2.VideoCapture(0)

# Load your custom trained YOLO model
# Make sure you downloaded 'best.pt' from Google Drive and placed it in a 'Weights' folder!
model = YOLO("Weights/best.pt")

# Define class names (Matches the classes from our dataset)
classNames = ['With Helmet', 'Without Helmet']

while True:
    success, img = cap.read()
    
    # If the video ends or cannot be read, break the loop
    if not success:
        print("Video ended or cannot be read.")
        break
        
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            
            # Draw the rectangle
            cvzone.cornerRect(img, (x1, y1, w, h))
            
            # Get confidence score and class index
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Display the label and confidence
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Show the image window
    cv2.imshow("Helmet Detection Live", img)
    
    # Press 'q' on your keyboard to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
