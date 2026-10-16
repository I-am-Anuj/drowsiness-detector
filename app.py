"""
SafeDrive AI - Real-Time Driver Drowsiness & Fatigue Detection System
Built with Streamlit, Streamlit-WebRTC, MediaPipe, and PyTorch (CNN & MobileNetV3)
Ready for cloud hosting on Streamlit Community Cloud and Hugging Face Spaces.
"""

import base64
import os
import threading
import time
from collections import deque
from datetime import datetime

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
from torchvision import models, transforms

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="SafeDrive AI | Driver Drowsiness Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Cyberpunk / Automotive Dashboard CSS
st.markdown(
    """
<style>
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0a0e14 100%);
        color: #e6edf3;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Header card */
    .header-card {
        background: rgba(22, 27, 34, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 14px;
        padding: 20px 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #388bfd, #79c0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-subtitle {
        color: #8b949e;
        font-size: 14px;
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b949e;
    }

    /* Status Pill */
    .status-badge-safe {
        background: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.5);
        padding: 12px 20px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
        animation: pulse-safe 2s infinite;
    }
    .status-badge-alert {
        background: rgba(248, 81, 73, 0.2);
        color: #f85149;
        border: 2px solid #f85149;
        padding: 12px 20px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: 800;
        text-align: center;
        animation: pulse-alert 0.8s infinite;
    }
    .status-badge-warn {
        background: rgba(210, 153, 34, 0.2);
        color: #e3b341;
        border: 1px solid rgba(210, 153, 34, 0.6);
        padding: 12px 20px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
    }

    @keyframes pulse-alert {
        0% { box-shadow: 0 0 0 0 rgba(248, 81, 73, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(248, 81, 73, 0); }
        100% { box-shadow: 0 0 0 0 rgba(248, 81, 73, 0); }
    }
    @keyframes pulse-safe {
        0% { box-shadow: 0 0 0 0 rgba(46, 160, 67, 0.4); }
        70% { box-shadow: 0 0 0 8px rgba(46, 160, 67, 0); }
        100% { box-shadow: 0 0 0 0 rgba(46, 160, 67, 0); }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load Alarm Audio for In-Browser Sound
# ---------------------------------------------------------
ALARM_FILE = "alarm.wav"
if os.path.exists(ALARM_FILE):
    with open(ALARM_FILE, "rb") as f:
        ALARM_BYTES = f.read()
        ALARM_B64 = base64.b64encode(ALARM_BYTES).decode("utf-8")
else:
    ALARM_BYTES = b""
    ALARM_B64 = ""

# ---------------------------------------------------------
# PyTorch Models & Architectures
# ---------------------------------------------------------
DEVICE = torch.device("cpu")
IMG_SIZE = 96
CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]


class DrowsinessCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
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


@st.cache_resource
def load_pytorch_models():
    cnn_model = None
    mobilenet_model = None

    if os.path.exists("cnn_baseline.pth"):
        try:
            cnn = DrowsinessCNN(num_classes=4)
            cnn.load_state_dict(torch.load("cnn_baseline.pth", map_location=DEVICE))
            cnn.eval()
            cnn_model = cnn
        except Exception as e:
            st.sidebar.warning(f"Failed to load CNN weights: {e}")

    if os.path.exists("mobilenet_finetuned.pth"):
        try:
            mb = models.mobilenet_v3_small(weights=None)
            mb.classifier[3] = nn.Linear(mb.classifier[3].in_features, 4)
            mb.load_state_dict(torch.load("mobilenet_finetuned.pth", map_location=DEVICE))
            mb.eval()
            mobilenet_model = mb
        except Exception as e:
            st.sidebar.warning(f"Failed to load MobileNet weights: {e}")

    return cnn_model, mobilenet_model


cnn_model, mobilenet_model = load_pytorch_models()

transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_crop(model, crop_bgr):
    """Run model prediction on a cropped BGR image."""
    if model is None or crop_bgr is None or crop_bgr.size == 0:
        return "Unknown", 0.0
    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor = transform_eval(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred_idx = probs.max(dim=1)
        return CLASS_NAMES[pred_idx.item()], float(conf.item())
    except Exception:
        return "Unknown", 0.0


# ---------------------------------------------------------
# MediaPipe Facial Landmark Math & Extraction
# ---------------------------------------------------------
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
TOP_LIP, BOTTOM_LIP = 13, 14
LEFT_MOUTH, RIGHT_MOUTH = 78, 308
MOUTH_BOX_IDX = [61, 291, 13, 14, 78, 308]


def euclidean_dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    pts = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    v1 = euclidean_dist(p2, p6)
    v2 = euclidean_dist(p3, p5)
    h = euclidean_dist(p1, p4)
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def calculate_mar(landmarks, frame_w, frame_h):
    top = (landmarks[TOP_LIP].x * frame_w, landmarks[TOP_LIP].y * frame_h)
    bottom = (landmarks[BOTTOM_LIP].x * frame_w, landmarks[BOTTOM_LIP].y * frame_h)
    left = (landmarks[LEFT_MOUTH].x * frame_w, landmarks[LEFT_MOUTH].y * frame_h)
    right = (landmarks[RIGHT_MOUTH].x * frame_w, landmarks[RIGHT_MOUTH].y * frame_h)
    v = euclidean_dist(top, bottom)
    h = euclidean_dist(left, right)
    if h == 0:
        return 0.0
    return v / h


def get_crop(frame, landmarks, indices, frame_w, frame_h, padding=15):
    """Crop square region around landmarks."""
    xs = [int(landmarks[i].x * frame_w) for i in indices]
    ys = [int(landmarks[i].y * frame_h) for i in indices]
    x1, x2 = min(xs) - padding, max(xs) + padding
    y1, y2 = min(ys) - padding, max(ys) + padding

    box_w, box_h = x2 - x1, y2 - y1
    side = max(box_w, box_h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1, x2 = cx - side // 2, cx + side // 2
    y1, y2 = cy - side // 2, cy + side // 2

    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame_w), min(y2, frame_h)

    if x2 <= x1 or y2 <= y1:
        return None, (0, 0, 0, 0)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ---------------------------------------------------------
# Shared State Container for Video Processor
# ---------------------------------------------------------
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "Normal"
        self.alert_active = False
        self.alert_reason = ""
        self.ear = 0.0
        self.mar = 0.0
        self.eye_state = "Open"
        self.mouth_state = "Normal"
        self.face_detected = False
        self.fps = 0.0
        self.event_log = []
        self.total_alerts = 0
        self.eye_closed_alerts = 0
        self.yawn_alerts = 0


if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()

shared_state = st.session_state.app_state

# ---------------------------------------------------------
# Sidebar Settings & Configuration
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/car.png", width=64)
st.sidebar.title("Configuration")

detection_mode = st.sidebar.selectbox(
    "Detection Engine",
    [
        "Hybrid (MediaPipe EAR/MAR + CNN Verification)",
        "MediaPipe Geometry Only (Ultra Fast)",
        "Custom CNN Classifier (cnn_baseline.pth)",
        "MobileNetV3 Classifier (mobilenet_finetuned.pth)",
    ],
    index=0,
    help="Select how eye closure and yawning are identified.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Sensitivity Controls")

ear_thresh = st.sidebar.slider(
    "Eye Aspect Ratio (EAR) Threshold",
    min_value=0.15,
    max_value=0.35,
    value=0.21,
    step=0.01,
    help="EAR below this value indicates closed eyes.",
)

ear_frames = st.sidebar.slider(
    "Consecutive Frames for Eye Closure",
    min_value=5,
    max_value=35,
    value=15,
    step=1,
    help="Number of consecutive frames eyes must remain closed before triggering alert.",
)

mar_thresh = st.sidebar.slider(
    "Mouth Aspect Ratio (MAR) Threshold",
    min_value=0.30,
    max_value=0.80,
    value=0.50,
    step=0.02,
    help="MAR above this value indicates yawning.",
)

mar_frames = st.sidebar.slider(
    "Consecutive Frames for Yawning",
    min_value=5,
    max_value=30,
    value=12,
    step=1,
    help="Number of consecutive frames mouth must remain wide before triggering alert.",
)

cnn_conf_thresh = st.sidebar.slider(
    "Deep Learning Confidence Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.60,
    step=0.05,
    help="Minimum probability required for CNN classification.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Visual Overlays")
show_landmarks = st.sidebar.checkbox("Show Eye & Mouth Meshes", value=True)
show_boxes = st.sidebar.checkbox("Show Crop Bounding Boxes", value=True)
show_hud = st.sidebar.checkbox("Show Telemetry HUD on Video", value=True)

with st.sidebar.expander("WebRTC & STUN / TURN Settings"):
    st.markdown("""
    **Active Public STUN Relays:**
    - `Google Public STUN (Port 19302)`
    - `Cloudflare STUN (Port 3478)`
    """)
    custom_metered_key = st.text_input(
        "Metered.ca API Key (Optional)",
        type="password",
        help="Optional: If your mobile network or corporate firewall blocks direct P2P UDP, enter a free Metered.ca API key for TURN relay traversal.",
    )
    if st.button("Test STUN Connectivity"):
        with st.spinner("Pinging STUN servers..."):
            import socket
            results = []
            for s_host, s_port in [("stun.l.google.com", 19302), ("stun.cloudflare.com", 3478)]:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.settimeout(2.0)
                    s.sendto(b"\x00\x01\x00\x00\x21\x12\xa4\x42" + b"\x00" * 12, (s_host, s_port))
                    data, _ = s.recvfrom(512)
                    results.append(f"[OK] {s_host}:{s_port} - Online ({len(data)}B)")
                except Exception as ex:
                    results.append(f"[FAIL] {s_host}:{s_port} - Failed ({ex})")
                finally:
                    s.close()
            st.success("STUN Diagnostics:\n" + "\n".join(results))

st.sidebar.markdown("---")
st.sidebar.subheader("Audio Alarm Controls")
enable_audio_alarm = st.sidebar.checkbox("Enable In-Browser Audio Alarm", value=True)
if st.sidebar.button("Test Audio Alarm"):
    if ALARM_BYTES:
        st.sidebar.audio(ALARM_BYTES, format="audio/wav", autoplay=True)
    st.sidebar.success("Audio alarm test triggered.")


# ---------------------------------------------------------
# Video Processor (WebRTC Frame Pipeline)
# ---------------------------------------------------------
class DrowsinessVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.eye_closed_counter = 0
        self.yawn_counter = 0
        self.was_alert = False

        self.eye_history = deque(maxlen=5)
        self.mouth_history = deque(maxlen=5)

        self.prev_time = time.time()
        self.frame_count = 0

        # Runtime adjustable attributes
        self.detection_mode = "Hybrid (MediaPipe EAR/MAR + CNN Verification)"
        self.ear_thresh = 0.21
        self.ear_frames = 15
        self.mar_thresh = 0.50
        self.mar_frames = 12
        self.cnn_conf_thresh = 0.60
        self.show_landmarks = True
        self.show_boxes = True
        self.show_hud = True

    def _process_frame(self, img):
        frame_h, frame_w = img.shape[:2]

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - self.prev_time, 1e-4)
        self.prev_time = curr_time

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        alert_reason = None
        ear = 0.0
        mar = 0.0
        eye_state = "Open"
        mouth_state = "Normal"
        face_detected = False

        active_dl_model = None
        if "Custom CNN" in self.detection_mode or "Hybrid" in self.detection_mode:
            active_dl_model = cnn_model
        elif "MobileNetV3" in self.detection_mode:
            active_dl_model = mobilenet_model

        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark

            # 1. Geometric EAR & MAR calculation
            left_ear = calculate_ear(landmarks, LEFT_EYE_IDX, frame_w, frame_h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_IDX, frame_w, frame_h)
            ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(landmarks, frame_w, frame_h)

            # 2. Extract crops for CNN / MobileNet if applicable
            left_crop, l_box = get_crop(img, landmarks, LEFT_EYE_IDX, frame_w, frame_h, padding=12)
            right_crop, r_box = get_crop(img, landmarks, RIGHT_EYE_IDX, frame_w, frame_h, padding=12)
            mouth_crop, m_box = get_crop(img, landmarks, MOUTH_BOX_IDX, frame_w, frame_h, padding=18)

            dl_eye_closed = False
            dl_yawning = False

            if active_dl_model is not None:
                eye_preds = []
                for c in [left_crop, right_crop]:
                    if c is not None and c.size > 0:
                        lbl, conf = classify_crop(active_dl_model, c)
                        if conf >= self.cnn_conf_thresh:
                            eye_preds.append(lbl)
                if eye_preds:
                    self.eye_history.append(max(set(eye_preds), key=eye_preds.count))
                smoothed_eye = max(set(self.eye_history), key=self.eye_history.count) if self.eye_history else "Open"
                if smoothed_eye == "Closed":
                    dl_eye_closed = True

                if mouth_crop is not None and mouth_crop.size > 0:
                    lbl, conf = classify_crop(active_dl_model, mouth_crop)
                    if conf >= self.cnn_conf_thresh:
                        self.mouth_history.append(lbl)
                smoothed_mouth = max(set(self.mouth_history), key=self.mouth_history.count) if self.mouth_history else "no_yawn"
                if smoothed_mouth == "yawn":
                    dl_yawning = True

            # 3. Decision Logic based on selected mode
            if "Hybrid" in self.detection_mode:
                is_eye_closed = (ear < self.ear_thresh) or dl_eye_closed
                is_yawn = (mar > self.mar_thresh) or dl_yawning
            elif "MediaPipe Geometry" in self.detection_mode:
                is_eye_closed = ear < self.ear_thresh
                is_yawn = mar > self.mar_thresh
            else:  # Pure Deep Learning mode
                is_eye_closed = dl_eye_closed
                is_yawn = dl_yawning

            # State tracking
            eye_state = "Closed" if is_eye_closed else "Open"
            mouth_state = "Yawning" if is_yawn else "Normal"

            if is_eye_closed:
                self.eye_closed_counter += 1
                if self.eye_closed_counter >= self.ear_frames:
                    alert_reason = "Eyes Closed"
            else:
                self.eye_closed_counter = 0

            if is_yawn:
                self.yawn_counter += 1
                if self.yawn_counter >= self.mar_frames:
                    alert_reason = "Yawning"
            else:
                self.yawn_counter = 0

            # 4. Overlays & Annotations
            if self.show_landmarks:
                for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                    pt = (int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h))
                    cv2.circle(img, pt, 2, (0, 255, 255), -1)
                for idx in [TOP_LIP, BOTTOM_LIP, LEFT_MOUTH, RIGHT_MOUTH]:
                    pt = (int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h))
                    cv2.circle(img, pt, 2, (255, 0, 255), -1)

            if self.show_boxes and active_dl_model is not None:
                for box in [l_box, r_box]:
                    if box[2] > box[0] and box[3] > box[1]:
                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                if m_box[2] > m_box[0] and m_box[3] > m_box[1]:
                    cv2.rectangle(img, (m_box[0], m_box[1]), (m_box[2], m_box[3]), (255, 200, 0), 1)

        else:
            self.eye_closed_counter = 0
            self.yawn_counter = 0

        # 5. Alert Banner & Frame Glow
        if alert_reason:
            cv2.rectangle(img, (0, 0), (frame_w - 1, frame_h - 1), (0, 0, 255), 8)
            cv2.rectangle(img, (0, 0), (frame_w, 60), (0, 0, 200), -1)
            cv2.putText(
                img,
                f"DROWSINESS ALERT: {alert_reason.upper()}!",
                (20, 42),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif self.show_hud:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (frame_w, 45), (20, 24, 30), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

            hud_text = f"EAR: {ear:.2f} ({eye_state})  |  MAR: {mar:.2f} ({mouth_state})  |  FPS: {fps:.1f}"
            status_color = (0, 255, 127) if face_detected else (100, 100, 255)
            cv2.putText(
                img,
                hud_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2,
                cv2.LINE_AA,
            )

        # 6. Update Shared State
        with shared_state.lock:
            shared_state.ear = ear
            shared_state.mar = mar
            shared_state.eye_state = eye_state
            shared_state.mouth_state = mouth_state
            shared_state.face_detected = face_detected
            shared_state.fps = fps

            if alert_reason:
                shared_state.alert_active = True
                shared_state.alert_reason = alert_reason
                shared_state.status = f"Alert: {alert_reason}"

                if not self.was_alert:
                    now_str = datetime.now().strftime("%H:%M:%S")
                    shared_state.event_log.append({
                        "Time": now_str,
                        "Event": f"Drowsiness Alert: {alert_reason}",
                        "EAR": f"{ear:.3f}",
                        "MAR": f"{mar:.3f}",
                    })
                    shared_state.total_alerts += 1
                    if "Eyes" in alert_reason:
                        shared_state.eye_closed_alerts += 1
                    elif "Yawn" in alert_reason:
                        shared_state.yawn_alerts += 1
                    self.was_alert = True
            else:
                shared_state.alert_active = False
                shared_state.alert_reason = ""
                shared_state.status = "Normal" if face_detected else "No Face Detected"
                self.was_alert = False

        return img

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            processed_img = self._process_frame(img)
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception:
            import traceback
            traceback.print_exc()
            return frame


# ---------------------------------------------------------
# Static Image Analysis (for Snapshot Camera & Uploads)
# ---------------------------------------------------------
@st.cache_resource
def get_shared_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def process_single_image(img, det_mode, ear_th, mar_th, cnn_th, show_lm, show_bx, show_hd):
    face_mesh = get_shared_face_mesh()
    frame_h, frame_w = img.shape[:2]
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    alert_reason = None
    ear = 0.0
    mar = 0.0
    eye_state = "Open"
    mouth_state = "Normal"
    face_detected = False

    active_dl_model = None
    if "Custom CNN" in det_mode or "Hybrid" in det_mode:
        active_dl_model = cnn_model
    elif "MobileNetV3" in det_mode:
        active_dl_model = mobilenet_model

    if results.multi_face_landmarks:
        face_detected = True
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = calculate_ear(landmarks, LEFT_EYE_IDX, frame_w, frame_h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE_IDX, frame_w, frame_h)
        ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(landmarks, frame_w, frame_h)

        left_crop, l_box = get_crop(img, landmarks, LEFT_EYE_IDX, frame_w, frame_h, padding=12)
        right_crop, r_box = get_crop(img, landmarks, RIGHT_EYE_IDX, frame_w, frame_h, padding=12)
        mouth_crop, m_box = get_crop(img, landmarks, MOUTH_BOX_IDX, frame_w, frame_h, padding=18)

        dl_eye_closed = False
        dl_yawning = False

        if active_dl_model is not None:
            eye_preds = []
            for c in [left_crop, right_crop]:
                if c is not None and c.size > 0:
                    lbl, conf = classify_crop(active_dl_model, c)
                    if conf >= cnn_th:
                        eye_preds.append(lbl)
            if eye_preds and eye_preds[0] == "Closed":
                dl_eye_closed = True

            if mouth_crop is not None and mouth_crop.size > 0:
                lbl, conf = classify_crop(active_dl_model, mouth_crop)
                if conf >= cnn_th and lbl == "yawn":
                    dl_yawning = True

        if "Hybrid" in det_mode:
            is_eye_closed = (ear < ear_th) or dl_eye_closed
            is_yawn = (mar > mar_th) or dl_yawning
        elif "MediaPipe Geometry" in det_mode:
            is_eye_closed = ear < ear_th
            is_yawn = mar > mar_th
        else:
            is_eye_closed = dl_eye_closed
            is_yawn = dl_yawning

        eye_state = "Closed" if is_eye_closed else "Open"
        mouth_state = "Yawning" if is_yawn else "Normal"

        if is_eye_closed:
            alert_reason = "Eyes Closed"
        elif is_yawn:
            alert_reason = "Yawning"

        if show_lm:
            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                pt = (int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h))
                cv2.circle(img, pt, 2, (0, 255, 255), -1)
            for idx in [TOP_LIP, BOTTOM_LIP, LEFT_MOUTH, RIGHT_MOUTH]:
                pt = (int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h))
                cv2.circle(img, pt, 2, (255, 0, 255), -1)

        if show_bx and active_dl_model is not None:
            for box in [l_box, r_box]:
                if box[2] > box[0] and box[3] > box[1]:
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            if m_box[2] > m_box[0] and m_box[3] > m_box[1]:
                cv2.rectangle(img, (m_box[0], m_box[1]), (m_box[2], m_box[3]), (255, 200, 0), 1)

    if alert_reason:
        cv2.rectangle(img, (0, 0), (frame_w - 1, frame_h - 1), (0, 0, 255), 8)
        cv2.rectangle(img, (0, 0), (frame_w, 60), (0, 0, 200), -1)
        cv2.putText(
            img,
            f"DROWSINESS ALERT: {alert_reason.upper()}!",
            (20, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    elif show_hd:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, 45), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        hud_text = f"EAR: {ear:.2f} ({eye_state})  |  MAR: {mar:.2f} ({mouth_state})"
        status_color = (0, 255, 127) if face_detected else (100, 100, 255)
        cv2.putText(
            img,
            hud_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA,
        )

    return img, ear, mar, eye_state, mouth_state, face_detected, alert_reason


# ---------------------------------------------------------
# In-Browser Audio Siren Component
# ---------------------------------------------------------
def render_audio_siren(alert_active: bool, alert_reason: str, enabled: bool = True):
    if not enabled:
        return

    active_js_val = "true" if alert_active else "false"
    bg_style = "rgba(248, 81, 73, 0.25)" if alert_active else "rgba(22, 27, 34, 0.7)"
    border_style = "#f85149" if alert_active else "rgba(48, 54, 61, 0.6)"
    text_color = "#ff7b72" if alert_active else "#8b949e"
    pulse_anim = "pulse-siren 0.8s infinite" if alert_active else "none"
    title_text = f"ALERT: {alert_reason.upper()}!" if alert_active else "Audio Alarm: Armed & Active"
    btn_text = "Mute Alarm" if alert_active else "Test Alarm Sound"
    btn_bg = "#f85149" if alert_active else "#238636"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #e6edf3;
            overflow: hidden;
        }}
        .siren-box {{
            background: {bg_style};
            border: 1px solid {border_style};
            border-radius: 10px;
            padding: 8px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            animation: {pulse_anim};
            box-sizing: border-box;
            height: 46px;
        }}
        @keyframes pulse-siren {{
            0% {{ box-shadow: 0 0 0 0 rgba(248, 81, 73, 0.7); }}
            70% {{ box-shadow: 0 0 0 10px rgba(248, 81, 73, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(248, 81, 73, 0); }}
        }}
        .siren-title {{
            font-size: 13px;
            font-weight: 700;
            color: {text_color};
            letter-spacing: 0.5px;
        }}
        .siren-btn {{
            background: {btn_bg};
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .siren-btn:hover {{
            filter: brightness(1.15);
        }}
    </style>
    </head>
    <body>
        <div class="siren-box">
            <span class="siren-title">{title_text}</span>
            <button id="sirenBtn" class="siren-btn" onclick="handleClick()">{btn_text}</button>
        </div>

        <audio id="alarmAudio" preload="auto" loop src="data:audio/wav;base64,{ALARM_B64}"></audio>

        <script>
            let muted = false;
            const audio = document.getElementById('alarmAudio');
            const alertActive = {active_js_val};

            function playWebAudioChime() {{
                try {{
                    const AudioContext = window.AudioContext || window.webkitAudioContext;
                    if (!AudioContext) return;
                    const ctx = new AudioContext();
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = 'sawtooth';
                    osc.frequency.setValueAtTime(900, ctx.currentTime);
                    osc.frequency.exponentialRampToValueAtTime(1400, ctx.currentTime + 0.25);
                    osc.frequency.exponentialRampToValueAtTime(900, ctx.currentTime + 0.5);
                    gain.gain.setValueAtTime(0.4, ctx.currentTime);
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.start();
                    osc.stop(ctx.currentTime + 0.5);
                }} catch(e) {{
                    console.error('AudioContext error:', e);
                }}
            }}

            function handleClick() {{
                if (alertActive) {{
                    muted = true;
                    if (audio) audio.pause();
                    document.getElementById('sirenBtn').innerText = 'MUTED';
                    document.getElementById('sirenBtn').style.background = '#6e7681';
                }} else {{
                    if (audio) {{
                        audio.currentTime = 0;
                        audio.play().then(() => {{
                            document.getElementById('sirenBtn').innerText = 'Audio Ready';
                            document.getElementById('sirenBtn').style.background = '#1f6feb';
                            setTimeout(() => {{
                                audio.pause();
                                audio.currentTime = 0;
                            }}, 1000);
                        }}).catch(() => {{
                            playWebAudioChime();
                            document.getElementById('sirenBtn').innerText = 'Synth Ready';
                        }});
                    }} else {{
                        playWebAudioChime();
                    }}
                }}
            }}

            if (alertActive && !muted) {{
                if (audio) {{
                    audio.play().catch(e => {{
                        console.log('Audio autoplay prevented, using WebAudio fallback:', e);
                        playWebAudioChime();
                    }});
                }} else {{
                    playWebAudioChime();
                }}
            }} else {{
                if (audio && !audio.paused) {{
                    audio.pause();
                    audio.currentTime = 0;
                }}
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=52)


# ---------------------------------------------------------
# WebRTC Configuration (Reliable STUN & Optional TURN servers)
# ---------------------------------------------------------
def get_rtc_configuration(metered_key: str = ""):
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:stun.cloudflare.com:3478"]},
    ]

    key = metered_key.strip()
    if not key:
        try:
            key = str(st.secrets.get("METERED_API_KEY", "")).strip()
        except Exception:
            key = ""
    if not key:
        key = os.environ.get("METERED_API_KEY", "").strip()

    if key:
        try:
            import json
            import urllib.request
            req = urllib.request.Request(
                f"https://metered.ca/api/v1/turn/credentials?apiKey={key}",
                headers={"User-Agent": "SafeDriveAI-Streamlit"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, list):
                    ice_servers.extend(data)
        except Exception:
            pass

    return RTCConfiguration({"iceServers": ice_servers})


# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown(
    """
<div class="header-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="header-title">SafeDrive AI Driver Monitoring System</h1>
            <p class="header-subtitle">Real-time Computer Vision & Deep Learning Fatigue Detection via In-Browser Webcam</p>
        </div>
        <div>
            <span style="background: rgba(88, 166, 255, 0.15); color: #58a6ff; border: 1px solid rgba(88, 166, 255, 0.4); padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">
                Cloud Hosted WebRTC
            </span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

col_video, col_telemetry = st.columns([1.35, 1.0], gap="medium")

with col_video:
    st.subheader("Driver Camera Feed")

    camera_mode = st.radio(
        "Camera Mode",
        ["Live WebRTC Camera", "Snapshot Photo Camera", "Upload Image File"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if camera_mode == "Live WebRTC Camera":
        rtc_cfg = get_rtc_configuration(custom_metered_key)
        webrtc_ctx = webrtc_streamer(
            key="drowsiness-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_cfg,
            video_processor_factory=DrowsinessVideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 30},
                },
                "audio": False,
            },
            video_html_attrs={
                "autoPlay": True,
                "controls": False,
                "style": {"width": "100%", "borderRadius": "10px"},
                "muted": True,
            },
            async_processing=True,
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.detection_mode = detection_mode
            webrtc_ctx.video_processor.ear_thresh = ear_thresh
            webrtc_ctx.video_processor.ear_frames = ear_frames
            webrtc_ctx.video_processor.mar_thresh = mar_thresh
            webrtc_ctx.video_processor.mar_frames = mar_frames
            webrtc_ctx.video_processor.cnn_conf_thresh = cnn_conf_thresh
            webrtc_ctx.video_processor.show_landmarks = show_landmarks
            webrtc_ctx.video_processor.show_boxes = show_boxes
            webrtc_ctx.video_processor.show_hud = show_hud

        if webrtc_ctx.state.playing:
            st.markdown("<span style='color: #3fb950; font-weight: 600;'>Camera connected and active</span>", unsafe_allow_html=True)
        elif webrtc_ctx.state.signalling:
            st.markdown("<span style='color: #e3b341; font-weight: 600;'>Negotiating WebRTC connection. Please allow camera permissions if prompted.</span>", unsafe_allow_html=True)
        else:
            st.caption("Click **START** above to launch the webcam feed. If prompted, allow camera access in your browser.")

        with st.expander("Camera Troubleshooting & Help"):
            st.markdown("""
            - **Camera shows black screen or loading?**
              1. **Check browser permissions**: Click the site settings icon in your browser address bar and verify **Camera** is set to **Allow**.
              2. **Check device lock**: Ensure no other application (Zoom, Teams, another tab) is actively using your camera.
              3. **Network NAT / Firewall blockage**: If your corporate firewall or cellular network blocks direct WebRTC UDP peer connections, switch to **'Snapshot Photo Camera'** above for zero-firewall instant detection.
            """)

    elif camera_mode == "Snapshot Photo Camera":
        st.caption("Instant capture mode works reliably on any browser and network without WebRTC firewall restrictions.")
        photo = st.camera_input("Capture Driver Snapshot")
        if photo is not None:
            bytes_data = photo.getvalue()
            img_arr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if img_arr is not None:
                proc_img, ear, mar, eye_state, mouth_state, face_detected, alert_reason = process_single_image(
                    img_arr, detection_mode, ear_thresh, mar_thresh, cnn_conf_thresh, show_landmarks, show_boxes, show_hud
                )
                with shared_state.lock:
                    shared_state.ear = ear
                    shared_state.mar = mar
                    shared_state.eye_state = eye_state
                    shared_state.mouth_state = mouth_state
                    shared_state.face_detected = face_detected
                    shared_state.fps = 30.0
                    if alert_reason:
                        shared_state.alert_active = True
                        shared_state.alert_reason = alert_reason
                        shared_state.status = f"Alert: {alert_reason}"
                        shared_state.total_alerts += 1
                        shared_state.event_log.append({
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Event": f"Drowsiness Alert: {alert_reason}",
                            "EAR": f"{ear:.3f}",
                            "MAR": f"{mar:.3f}",
                        })
                    else:
                        shared_state.alert_active = False
                        shared_state.alert_reason = ""
                        shared_state.status = "Normal" if face_detected else "No Face Detected"

                rgb_display = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
                st.image(rgb_display, caption="Analyzed Driver Snapshot", use_container_width=True)

    else:  # Upload Image
        uploaded_file = st.file_uploader("Upload driver face image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            img_arr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if img_arr is not None:
                proc_img, ear, mar, eye_state, mouth_state, face_detected, alert_reason = process_single_image(
                    img_arr, detection_mode, ear_thresh, mar_thresh, cnn_conf_thresh, show_landmarks, show_boxes, show_hud
                )
                with shared_state.lock:
                    shared_state.ear = ear
                    shared_state.mar = mar
                    shared_state.eye_state = eye_state
                    shared_state.mouth_state = mouth_state
                    shared_state.face_detected = face_detected
                    if alert_reason:
                        shared_state.alert_active = True
                        shared_state.alert_reason = alert_reason
                        shared_state.status = f"Alert: {alert_reason}"
                    else:
                        shared_state.alert_active = False
                        shared_state.alert_reason = ""
                        shared_state.status = "Normal" if face_detected else "No Face Detected"

                rgb_display = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
                st.image(rgb_display, caption="Analyzed Image", use_container_width=True)


with col_telemetry:
    st.subheader("Driver Telemetry & Status")

    # Reactive metrics fragment refreshing every 1.5 seconds without re-rendering the WebRTC video
    @st.fragment(run_every=1.5)
    def render_live_telemetry():
        with shared_state.lock:
            status = shared_state.status
            alert_active = shared_state.alert_active
            alert_reason = shared_state.alert_reason
            ear = shared_state.ear
            mar = shared_state.mar
            eye_state = shared_state.eye_state
            mouth_state = shared_state.mouth_state
            face_detected = shared_state.face_detected
            fps = shared_state.fps
            total_alerts = shared_state.total_alerts
            events = list(reversed(shared_state.event_log[-10:]))

        # Audio Siren Alert Component (plays automatically on drowsiness)
        render_audio_siren(alert_active, alert_reason, enable_audio_alarm)

        # Status Badge
        if alert_active:
            st.markdown(
                f"""
            <div class="status-badge-alert">
                DROWSINESS DETECTED: {alert_reason.upper()}
            </div>
            """,
                unsafe_allow_html=True,
            )
            # Dual redundancy: native Streamlit audio player with autoplay
            if enable_audio_alarm and ALARM_BYTES:
                st.audio(ALARM_BYTES, format="audio/wav", loop=True, autoplay=True)
        elif not face_detected:
            st.markdown(
                """
            <div class="status-badge-warn">
                NO FACE DETECTED
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="status-badge-safe">
                DRIVER ALERT & ATTENTIVE
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # 4 Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">EAR Ratio</div>
                <div class="metric-val" style="color: {'#f85149' if ear < ear_thresh and face_detected else '#3fb950'};">
                    {ear:.2f}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">MAR Ratio</div>
                <div class="metric-val" style="color: {'#f85149' if mar > mar_thresh and face_detected else '#3fb950'};">
                    {mar:.2f}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Eye State</div>
                <div class="metric-val" style="color: {'#f85149' if eye_state == 'Closed' else '#58a6ff'};">
                    {eye_state}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Mouth State</div>
                <div class="metric-val" style="color: {'#f85149' if mouth_state == 'Yawning' else '#58a6ff'};">
                    {mouth_state}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Progress Bars for EAR and MAR
        st.caption(f"Eye Aspect Ratio (Threshold: {ear_thresh:.2f})")
        st.progress(min(max(ear / 0.5, 0.0), 1.0))

        st.caption(f"Mouth Aspect Ratio (Threshold: {mar_thresh:.2f})")
        st.progress(min(max(mar / 1.0, 0.0), 1.0))

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Event Log Table
        st.markdown("##### Recent Detection Events")
        if events:
            df_log = pd.DataFrame(events)
            st.dataframe(df_log, use_container_width=True, hide_index=True)
        else:
            st.info("No fatigue events recorded yet. Drive safely!")

        # Analytics Footer
        st.caption(f"Total Session Alerts: **{total_alerts}** | Pipeline FPS: **{fps:.1f}**")

    render_live_telemetry()

# ---------------------------------------------------------
# Model Performance & Reference Section
# ---------------------------------------------------------
st.markdown("---")
with st.expander("Model Architecture & Benchmark Metrics"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Custom CNN Baseline**")
        st.markdown("""
        - **Input**: 96x96 BGR Crop
        - **Architecture**: 3x Conv2D (32, 64, 128) + ReLU + MaxPool2d + Dropout(0.4) + FC(128, 4)
        - **Accuracy**: ~96.5% on test split
        """)
        if os.path.exists("confusion_matrix_custom_cnn.png"):
            st.image("confusion_matrix_custom_cnn.png", caption="Custom CNN Confusion Matrix")

    with c2:
        st.markdown("**MobileNetV3-Small (Fine-Tuned)**")
        st.markdown("""
        - **Input**: 96x96 BGR Crop
        - **Backbone**: ImageNet Pretrained MobileNetV3-Small (Top 3 layers fine-tuned)
        - **Accuracy**: ~98.1% on test split
        """)
        if os.path.exists("confusion_matrix_mobilenetv3_small.png"):
            st.image("confusion_matrix_mobilenetv3_small.png", caption="MobileNetV3 Confusion Matrix")

    with c3:
        st.markdown("**MediaPipe Face Mesh (478 Landmarks)**")
        st.markdown(r"""
        - **Topology**: Blazeface + 478 3D facial landmarks
        - **EAR Formula**: $(||p2 - p6|| + ||p3 - p5||) / (2 \cdot ||p1 - p4||)$
        - **MAR Formula**: $||top - bottom|| / ||left - right||$
        - **Inference Time**: < 10ms CPU latency
        """)
        if os.path.exists("model_comparison.png"):
            st.image("model_comparison.png", caption="Model Comparison Metric")
