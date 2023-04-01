import cv2
import numpy as np

# Load the image
img = cv2.imread("image.jpg")

# Preprocess the image
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

# Load the YOLO model
net = cv2.dnn.readNetFromDarknet("yolov4-tiny-custom.cfg", "yolov4-tiny-custom_best.weights")

# Perform inference
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

# Create the binary mask
mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            width = int(detection[2] * img.shape[1])
            height = int(detection[3] * img.shape[0])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            # Set the corresponding pixels in the mask to 1
            mask[top:top+height, left:left+width] = 1

# Apply segmentation
segmented = cv2.merge([mask*img[:,:,0], mask*img[:,:,1], mask*img[:,:,2]])

# Display the result
cv2.imshow("Result", segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()