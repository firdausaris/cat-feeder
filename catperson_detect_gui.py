import cv2
import numpy as np

# Load the MobileNetSSD model
net = cv2.dnn.readNetFromCaffe(
    "model/MobileNetSSD_deploy.prototxt",
    "model/MobileNetSSD_deploy.caffemodel"
)

# Class labels known to the model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Objects we want to detect
TARGETS = {"cat", "person"}

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🟢 Starting Cat & Person Detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("❌ Empty frame")
        continue

    # Prepare the frame for prediction
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Analyze detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            if label in TARGETS:
                box = detections[0, 0, i, 3:7] * np.array(
                    [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                color = (0, 255, 0) if label == "person" else (0, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                text = f"{label}: {int(confidence * 100)}%"
                cv2.putText(frame, text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Cat & Person Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
