import cv2
import numpy as np
from tkinter import *



# Load the YOLO model
model = cv2.dnn.readNetFromDarknet("yolov4-tiny-custom.cfg", "yolov4-tiny-custom_best.weights")
# Load class labels
classes = []
with open('obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set confidence threshold
confidence_threshold = 0.4

# Set non-maximum suppression threshold
nms_threshold = 0.4

# Load video file
cap = cv2.VideoCapture('video.mp4')
object_detected = False


object_detected = False
while cap.isOpened():
    # Read a frame from the video file
    ret, frame = cap.read()
    
    # Check if the frame was successfully read
    if not ret:
        break
    
    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    

    # Set the input for the model
    model.setInput(blob)
    
    # Get the output from the model
    output = model.forward()
    
    # Extract bounding boxes and confidence scores
    boxes = []
    confidences = []
    class_ids = []
    height, width = frame.shape[:2]
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confidence_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    
    # Apply non-maximum suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    # Draw bounding boxes and labels on the frame
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    
    if len(indices) > 0 and not object_detected:
        
        object_detected = True  # Set object_detected flag to True
        print("envio señal")
    
    cv2.imshow('Video', frame)
    
    
    if len(indices) == 0:
        object_detected = False # Set object_detected flag to True
    # Check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

