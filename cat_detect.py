import cv2
import time
from gpiozero import LED

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(
    "model/MobileNetSSD_deploy.prototxt",
    "model/MobileNetSSD_deploy.caffemodel"
)

# Class labels from COCO (class 17 is "cat")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# GPIO setup
actuator = LED(17)
last_triggered = 0
cooldown = 10  # seconds

# Start video capture
cap = cv2.VideoCapture(0)
print("🐾 Cat Detection System Running. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera frame not received.")
            break

        # Prepare frame for MobileNet SSD
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        cat_found = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            if confidence > 0.5 and CLASSES[class_id] == "cat":

                cat_found = True
                break

        if cat_found and (time.time() - last_triggered > cooldown):
            print("😺 Cat detected! Activating actuator.")
            actuator.on()
            time.sleep(2)
            actuator.off()
            last_triggered = time.time()
        else:
            print("🔍 No cat detected.")

        time.sleep(1)

except KeyboardInterrupt:
    print("🛑 Detection stopped by user.")

finally:
    cap.release()
