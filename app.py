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
import torch
import torch.nn as nn
from PIL import Image
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
from torchvision import models, transforms

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="SafeDrive AI | Driver Drowsiness Detector",
    page_icon="🚗",
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
        ALARM_B64 = base64.b64encode(f.read()).decode("utf-8")
else:
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


# ---------------------------------------------------------
# Video Processor (WebRTC Frame Pipeline)
# ---------------------------------------------------------
class DrowsinessVideoProcessor:
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

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        frame_h, frame_w = img.shape[:2]

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - self.prev_time, 1e-4)
        self.prev_time = curr_time

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        alert_reason = None
        ear = 0.0
        mar = 0.0
        eye_state = "Open"
        mouth_state = "Normal"
        face_detected = False

        active_dl_model = None
        if "Custom CNN" in detection_mode or "Hybrid" in detection_mode:
            active_dl_model = cnn_model
        elif "MobileNetV3" in detection_mode:
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
                # Classify eyes
                eye_preds = []
                for c in [left_crop, right_crop]:
                    if c is not None and c.size > 0:
                        lbl, conf = classify_crop(active_dl_model, c)
                        if conf >= cnn_conf_thresh:
                            eye_preds.append(lbl)
                if eye_preds:
                    self.eye_history.append(max(set(eye_preds), key=eye_preds.count))
                smoothed_eye = max(set(self.eye_history), key=self.eye_history.count) if self.eye_history else "Open"
                if smoothed_eye == "Closed":
                    dl_eye_closed = True

                # Classify mouth
                if mouth_crop is not None and mouth_crop.size > 0:
                    lbl, conf = classify_crop(active_dl_model, mouth_crop)
                    if conf >= cnn_conf_thresh:
                        self.mouth_history.append(lbl)
                smoothed_mouth = max(set(self.mouth_history), key=self.mouth_history.count) if self.mouth_history else "no_yawn"
                if smoothed_mouth == "yawn":
                    dl_yawning = True

            # 3. Decision Logic based on selected mode
            if "Hybrid" in detection_mode:
                is_eye_closed = (ear < ear_thresh) or dl_eye_closed
                is_yawn = (mar > mar_thresh) or dl_yawning
            elif "MediaPipe Geometry" in detection_mode:
                is_eye_closed = ear < ear_thresh
                is_yawn = mar > mar_thresh
            else:  # Pure Deep Learning mode
                is_eye_closed = dl_eye_closed
                is_yawn = dl_yawning

            # State tracking
            eye_state = "Closed" if is_eye_closed else "Open"
            mouth_state = "Yawning" if is_yawn else "Normal"

            if is_eye_closed:
                self.eye_closed_counter += 1
                if self.eye_closed_counter >= ear_frames:
                    alert_reason = "Eyes Closed"
            else:
                self.eye_closed_counter = 0

            if is_yawn:
                self.yawn_counter += 1
                if self.yawn_counter >= mar_frames:
                    alert_reason = "Yawning"
            else:
                self.yawn_counter = 0

            # 4. Overlays & Annotations
            if show_landmarks:
                for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                    pt = (int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h))
                    cv2.circle(img, pt, 2, (0, 255, 255), -1)
                for idx in [TOP_LIP, BOTTOM_LIP, LEFT_MOUTH, RIGHT_MOUTH]:
                    pt = (int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h))
                    cv2.circle(img, pt, 2, (255, 0, 255), -1)

            if show_boxes and active_dl_model is not None:
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
            # Red warning border
            cv2.rectangle(img, (0, 0), (frame_w - 1, frame_h - 1), (0, 0, 255), 8)
            # Top warning banner
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
        elif show_hud:
            # Top translucent HUD bar
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

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------
# WebRTC Configuration (STUN servers for cloud deployment)
# ---------------------------------------------------------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})

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
    st.subheader("📹 Real-Time In-Browser Camera")

    # In-Browser Audio Player & Siren Trigger
    st.markdown(
        f"""
    <div style="background: rgba(22, 27, 34, 0.7); border: 1px solid rgba(48, 54, 61, 0.6); padding: 10px 16px; border-radius: 10px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
        <span style="font-size: 13px; color: #8b949e;">🔊 Browser Audio Alarm</span>
        <button id="sirenBtn" onclick="testAlarm()" style="background: #238636; color: white; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600;">
            Arm / Test Audio Siren
        </button>
    </div>

    <audio id="alarmAudio" preload="auto" loop>
        <source src="data:audio/wav;base64,{ALARM_B64}" type="audio/wav">
    </audio>

    <script>
    let audioArmed = false;
    const alarm = document.getElementById('alarmAudio');

    function testAlarm() {{
        if (!alarm) return;
        alarm.currentTime = 0;
        alarm.play().then(() => {{
            audioArmed = true;
            document.getElementById('sirenBtn').innerText = 'Audio Armed & Ready ✓';
            document.getElementById('sirenBtn').style.background = '#1f6feb';
            setTimeout(() => {{
                alarm.pause();
                alarm.currentTime = 0;
            }}, 800);
        }}).catch(err => {{
            console.log('Audio autoplay prevented:', err);
        }});
    }}

    window.triggerBrowserAlarm = function(play) {{
        if (!alarm) return;
        if (play) {{
            if (alarm.paused) {{
                alarm.play().catch(e => console.log('Autoplay policy error:', e));
            }}
        }} else {{
            if (!alarm.paused) {{
                alarm.pause();
                alarm.currentTime = 0;
            }}
        }}
    }};
    </script>
    """,
        unsafe_allow_html=True,
    )

    webrtc_ctx = webrtc_streamer(
        key="drowsiness-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=DrowsinessVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.caption("Click **START** to open your webcam stream. Works on mobile, tablet, and desktop.")


with col_telemetry:
    st.subheader("📊 Driver Telemetry & Status")

    # Reactive metrics fragment refreshing every 1 second without re-rendering the WebRTC video
    @st.fragment(run_every=1.0)
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

        # Status Badge
        if alert_active:
            st.markdown(
                f"""
            <div class="status-badge-alert">
                ⚠️ DROWSINESS DETECTED: {alert_reason.upper()}
            </div>
            <script>
            if (window.triggerBrowserAlarm) window.triggerBrowserAlarm(true);
            </script>
            """,
                unsafe_allow_html=True,
            )
        elif not face_detected:
            st.markdown(
                """
            <div class="status-badge-warn">
                👤 NO FACE DETECTED
            </div>
            <script>
            if (window.triggerBrowserAlarm) window.triggerBrowserAlarm(false);
            </script>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="status-badge-safe">
                ✓ DRIVER ALERT & ATTENTIVE
            </div>
            <script>
            if (window.triggerBrowserAlarm) window.triggerBrowserAlarm(false);
            </script>
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
        st.markdown("##### 📜 Recent Detection Event Log")
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
with st.expander("📈 Model Architecture & Training Evaluation Benchmarks"):
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
