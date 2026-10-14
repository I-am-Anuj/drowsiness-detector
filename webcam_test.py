import cv2
import time

# Use V4L2 backend on Linux for better compatibility
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")
time.sleep(1)  # Warm-up delay

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Dropped frame, retrying...")
        continue

    cv2.imshow("Webcam Feed", frame)

    # 30ms delay gives the GUI backend time to render
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()