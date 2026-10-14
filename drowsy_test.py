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

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.21       # below this = eyes considered "closed"
CONSEC_FRAMES = 30         # how many frames in a row before we call it drowsy

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    points = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")
time.sleep(1)

closed_frame_counter = 0  # tracks consecutive low-EAR frames

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

        if avg_ear < EAR_THRESHOLD:
            closed_frame_counter += 1
            if closed_frame_counter >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            closed_frame_counter = 0  # reset — eyes are open again

    else:
        cv2.putText(frame, "No face detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        closed_frame_counter = 0

    cv2.imshow("Drowsiness Test", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()