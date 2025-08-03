import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No frame")
        break

    display = cv2.resize(frame, (320, 240))
    cv2.imshow("Cat Detector", display)

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
