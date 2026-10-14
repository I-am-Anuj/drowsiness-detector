import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import threading
from datetime import datetime
from flask import Flask, render_template

# ---------- Shared state between Flask thread and camera thread ----------
current_status = "Normal"
event_log = []  # list of {"time": ..., "event": ...}
log_lock = threading.Lock()

# ---------- Flask app ----------
app = Flask(__name__)

@app.route("/")
def dashboard():
    with log_lock:
        recent_log = list(reversed(event_log[-20:]))  # newest first, last 20 events
    return render_template("dashboard.html", status=current_status, log=recent_log)

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# ---------- MediaPipe / detection setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
TOP_LIP, BOTTOM_LIP = 13, 14
LEFT_MOUTH, RIGHT_MOUTH = 78, 308

EAR_THRESHOLD = 0.21
EAR_CONSEC_FRAMES = 20
MAR_THRESHOLD = 0.5
MAR_CONSEC_FRAMES = 15

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    points = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def calculate_mar(landmarks, frame_w, frame_h):
    top = (landmarks[TOP_LIP].x * frame_w, landmarks[TOP_LIP].y * frame_h)
    bottom = (landmarks[BOTTOM_LIP].x * frame_w, landmarks[BOTTOM_LIP].y * frame_h)
    left = (landmarks[LEFT_MOUTH].x * frame_w, landmarks[LEFT_MOUTH].y * frame_h)
    right = (landmarks[RIGHT_MOUTH].x * frame_w, landmarks[RIGHT_MOUTH].y * frame_h)
    vertical = euclidean_dist(top, bottom)
    horizontal = euclidean_dist(left, right)
    return vertical / horizontal

def log_event(event_text):
    global current_status
    with log_lock:
        event_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "event": event_text
        })
    current_status = event_text

# ---------- Camera loop (runs in main thread) ----------
def run_detection():
    global current_status

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened. Dashboard at http://localhost:5000  |  Press 'q' to quit.")
    time.sleep(1)

    eye_closed_counter = 0
    yawn_counter = 0
    alarm_playing = False
    was_alert = False  # to detect the moment status changes, so we don't spam the log

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        alert_reason = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = calculate_ear(landmarks, LEFT_EYE, frame_w, frame_h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, frame_w, frame_h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                eye_closed_counter += 1
                if eye_closed_counter >= EAR_CONSEC_FRAMES:
                    alert_reason = "Eyes Closed"
            else:
                eye_closed_counter = 0

            mar = calculate_mar(landmarks, frame_w, frame_h)
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= MAR_CONSEC_FRAMES:
                    alert_reason = "Yawning"
            else:
                yawn_counter = 0

            cv2.putText(frame, f"EAR: {avg_ear:.3f}  MAR: {mar:.3f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            eye_closed_counter = 0
            yawn_counter = 0

        # ---------- Alarm + logging ----------
        if alert_reason:
            cv2.putText(frame, f"DROWSINESS ALERT: {alert_reason}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), (0, 0, 255), 10)

            if not alarm_playing:
                pygame.mixer.music.play(-1)
                alarm_playing = True

            if not was_alert:  # log only on the transition into alert state
                log_event(alert_reason)
                was_alert = True
        else:
            if alarm_playing:
                pygame.mixer.music.stop()
                alarm_playing = False
            if was_alert:
                current_status = "Normal"
                was_alert = False

        cv2.imshow("Drowsiness Detector", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()

# ---------- Entry point ----------
if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    run_detection()