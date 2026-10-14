import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices for each eye (6 points each)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    # Convert normalized landmark coords to actual pixel coords
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((lm.x * frame_w, lm.y * frame_h))

    # points order: [p1(left corner), p2, p3, p4(right corner), p5, p6]
    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")
time.sleep(1)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_h, frame_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = calculate_ear(landmarks, LEFT_EYE, frame_w, frame_h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, frame_w, frame_h)
        avg_ear = (left_ear + right_ear) / 2.0

        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("EAR Test", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()