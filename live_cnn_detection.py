import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pygame
import time
from collections import deque

DEVICE = torch.device("cpu")
IMG_SIZE = 96
CLASS_NAMES = ['Closed', 'Open', 'no_yawn', 'yawn']  # must match training order exactly

# ---------- Rebuild and load your trained CNN ----------
class DrowsinessCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = DrowsinessCNN(num_classes=4)
model.load_state_dict(torch.load("cnn_baseline.pth", map_location=DEVICE))
model.eval()

# Same normalization used during training — must match exactly
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_crop(crop_bgr):
    """Returns (predicted class name, confidence float 0-1)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = probs.max(dim=1)
    return CLASS_NAMES[pred_idx.item()], confidence.item()

# ---------- MediaPipe setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Landmark indices to build bounding boxes around eye/mouth regions
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MOUTH_IDX = [61, 291, 13, 14, 78, 308]

def get_crop(frame, landmarks, indices, frame_w, frame_h, padding=15):
    """Builds a SQUARE bounding box around given landmarks and returns the cropped region."""
    xs = [int(landmarks[i].x * frame_w) for i in indices]
    ys = [int(landmarks[i].y * frame_h) for i in indices]
    x1, x2 = min(xs) - padding, max(xs) + padding
    y1, y2 = min(ys) - padding, max(ys) + padding

    # Force a square box by expanding the shorter side to match the longer one
    box_w, box_h = x2 - x1, y2 - y1
    side = max(box_w, box_h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1, x2 = cx - side // 2, cx + side // 2
    y1, y2 = cy - side // 2, cy + side // 2

    # Clamp to frame boundaries
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame_w), min(y2, frame_h)

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

# ---------- Smoothing setup ----------
eye_history = deque(maxlen=5)    # last 5 eye predictions
mouth_history = deque(maxlen=5)  # last 5 mouth predictions

def smoothed_label(history):
    """Majority vote over recent frames — reduces flicker."""
    if not history:
        return "Unknown"
    return max(set(history), key=history.count)

# ---------- Alarm setup ----------
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

EYE_CLOSED_FRAMES = 15   # consecutive frames before alerting
YAWN_FRAMES = 10

cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # Drop cv2.CAP_V4L2 if on Windows
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")
time.sleep(1)

closed_counter = 0
yawn_counter = 0
alarm_playing = False

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

        left_eye_crop = get_crop(frame, landmarks, LEFT_EYE_IDX, frame_w, frame_h)
        right_eye_crop = get_crop(frame, landmarks, RIGHT_EYE_IDX, frame_w, frame_h)
        mouth_crop = get_crop(frame, landmarks, MOUTH_IDX, frame_w, frame_h, padding=20)

        eye_state, mouth_state = "Unknown", "Unknown"
        CONF_THRESHOLD = 0.6  # ignore low-confidence predictions instead of trusting a coin-flip

        # --- Eyes: classify both, only trust confident ones ---
        eye_preds = []
        for crop in [left_eye_crop, right_eye_crop]:
            if crop is not None and crop.size > 0:
                label, conf = classify_crop(crop)
                if conf >= CONF_THRESHOLD:
                    eye_preds.append(label)

        if eye_preds:
            eye_history.append(max(set(eye_preds), key=eye_preds.count))
        
        eye_state = smoothed_label(eye_history)
        
        if eye_state == "Closed":
            closed_counter += 1
            print(f"[DEBUG] eye_state={eye_state}  closed_counter={closed_counter}")
            if closed_counter >= EYE_CLOSED_FRAMES:
                alert_reason = "Eyes Closed"
        else:
            closed_counter = 0

        # --- Mouth ---
        if mouth_crop is not None and mouth_crop.size > 0:
            label, conf = classify_crop(mouth_crop)
            if conf >= CONF_THRESHOLD:
                mouth_history.append(label)
        
        mouth_state = smoothed_label(mouth_history)
        
        if mouth_state == "yawn":
            yawn_counter += 1
            if yawn_counter >= YAWN_FRAMES:
                alert_reason = "Yawning"
        else:
            yawn_counter = 0

        cv2.putText(frame, f"Eye: {eye_state}  Mouth: {mouth_state}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("[DEBUG] No face detected — resetting counters")
        closed_counter = 0
        yawn_counter = 0

    # Trigger or stop alarm based on logic
    if alert_reason:
        cv2.putText(frame, f"DROWSINESS ALERT: {alert_reason}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.rectangle(frame, (0, 0), (frame_w - 1, frame_h - 1), (0, 0, 255), 10)
        if not alarm_playing:
            pygame.mixer.music.play(-1)
            alarm_playing = True
            print("[DEBUG] play() called | busy:", pygame.mixer.music.get_busy(),
                  "| volume:", pygame.mixer.music.get_volume())
    else:
        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    cv2.imshow("CNN-Based Drowsiness Detector", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()