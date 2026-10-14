---
title: SafeDrive AI Driver Drowsiness Detector
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.39.1
app_file: app.py
pinned: false
---

# SafeDrive AI | Real-Time Driver Drowsiness & Fatigue Detector 🚗💤

An AI-powered computer vision application designed to monitor driver alertness in real time through an in-browser webcam stream. Combines **MediaPipe Face Mesh (478 landmarks)** with custom **PyTorch Deep Learning CNN** and fine-tuned **MobileNetV3** architectures to detect closed eyes, prolonged blinking, and yawning with high precision.

---

## 🌟 Key Features

- **In-Browser Webcam via WebRTC**: Seamless video streaming directly in the browser across laptops, tablets, and phones without needing server-attached webcams.
- **Hybrid AI Engine**:
  - **MediaPipe Geometric EAR & MAR**: Computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) with sub-10ms CPU latency.
  - **Custom CNN Classifier**: 4-class convolutional neural network (`cnn_baseline.pth`) classifying cropped eye and mouth regions.
  - **MobileNetV3 Fine-Tuned**: High-accuracy lightweight transfer learning model (`mobilenet_finetuned.pth`).
- **In-Browser Audio Alarm**: Triggers an alert siren (`alarm.wav`) directly in the user's browser/headphones when fatigue is detected.
- **Real-Time Telemetry HUD**: Live metrics dashboard displaying EAR, MAR, eye closure counters, yawn duration, pipeline FPS, and event logs.
- **Configurable Sensitivity**: Real-time sidebar sliders to adjust thresholds and trigger sensitivity on the fly.

---

## 🚀 Free Cloud Hosting Options

### Option 1: Streamlit Community Cloud (Recommended - 100% Free)

Deploy with zero configuration directly from your GitHub repository:

1. **Initialize Git & Push to GitHub**:
   ```bash
   cd ~/Documents/drowsiness-detector
   git init -b main
   git add .
   git commit -m "feat: complete cloud-hosted drowsiness detection app"
   
   # Create a repository on github.com, then link and push:
   git remote add origin https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
   git push -u origin main
   ```
   *(Note: `.gitignore` is already configured to exclude the large dataset and zip files so your push stays fast and lightweight).*

2. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **"New app"**.
   - Select your repository: `<YOUR_USERNAME>/<REPO_NAME>`.
   - Branch: `main`.
   - Main file path: `app.py`.
   - Click **"Deploy!"**.
   - Your app will be live with a public HTTPS URL (e.g. `https://<your-app>.streamlit.app`).

---

### Option 2: Hugging Face Spaces (Free Cloud Hosting with WebRTC)

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **Spaces** > **Create new Space**.
3. Name your space (e.g. `driver-drowsiness-detector`).
4. Select **Streamlit** as the Space SDK.
5. Choose **Public** (Free CPU tier).
6. Push your code to the Hugging Face Git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/driver-drowsiness-detector
   git push hf main
   ```
7. Hugging Face will automatically read the YAML frontmatter in `README.md`, install dependencies, and launch your app!

---

### Option 3: Docker / Render / Railway / Cloud Run

You can deploy the included [`Dockerfile`](file:///home/anuj/Documents/drowsiness-detector/Dockerfile) to any container platform:

```bash
# Build Docker image
docker build -t drowsiness-detector .

# Run container locally
docker run -p 8501:8501 drowsiness-detector
```

On **Render** or **Railway**:
- Connect your GitHub repository.
- Select **Web Service** with Docker environment or Python environment (`streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`).

---

## 💻 Local Quickstart

To run and test the app locally on your machine:

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Start the Streamlit application
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── app.py                             # Main Streamlit WebRTC application
├── main.py                            # Original Flask + local desktop OpenCV detector
├── live_cnn_detection.py              # CNN desktop inference script
├── cnn_baseline.pth                   # Trained custom CNN model weights (~9.8 MB)
├── mobilenet_finetuned.pth            # Fine-tuned MobileNetV3 model weights (~6.2 MB)
├── alarm.wav                          # Alert sound effect
├── requirements.txt                   # Production-pinned python dependencies
├── packages.txt                       # System dependencies for cloud deployment
├── Dockerfile                         # Container build definition
├── .gitignore                         # Git exclusion rules for large datasets
└── templates/
    └── dashboard.html                 # Flask dashboard template
```
