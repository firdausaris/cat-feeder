import cv2
import time
from gpiozero import LED

# ==== Setup paths ====
MODEL_PROTO = "model/MobileNetSSD_deploy.prototxt"
MODEL_WEIGHTS = "model/MobileNetSSD_deploy.caffemodel"

# ==== Load model ====
net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)

# ==== Labels ====
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# ==== GPIO setup ====
led = LED(17)
cooldown_seconds = 10
last_triggered_time = 0

TARGET = "cat"

# ==== Open camera ====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)

print("📷 Camera resolution:", cap.get(3), "x", cap.get(4))
print("😺 Starting real-time cat detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("❌ Empty frame")
        continue

    # Resize to 300x300 for DNN input
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected_cat = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if confidence > 0.5 and CLASSES[idx] == "cat":
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], fr>
            (startX, startY, endX, endY) = box.astype("int")
            label = f"Cat: {confidence * 100:.1f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detected_cat = True

    # Trigger actuator if cat detected and cooldown passed
    if detected_cat and (time.time() - last_triggered_time) > cooldown_seconds:
        print("😺 Cat detected! Triggering actuator.")
        led.on()
        time.sleep(1)
        led.off()
        last_triggered_time = time.time()

    # Display the frame
    cv2.imshow("Cat Detector", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
