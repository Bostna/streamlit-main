# WhenAISeesLitterStreamlit  

**Real-time litter detection with Streamlit & YOLO**  
A simple and interactive web app to detect litter using a local YOLO model via Streamlit.

---

## Demo

<p align="center">
  <img src="Bostna/whenaiseeslitterstreamlit/logo.png" alt="Project Logo" width="150"/>
</p>


---

## Features

Two input modes: Upload image and Camera
Loads weights from a public URL or a local file
Adjustable thresholds: base confidence, IoU, per-class minimums, minimum box area, inference image size, optional TTA
Disposal guidance cards for Shibuya with official links and images
SDG tiles: 11, 12, 13
Live video was removed

---

## Contents

| File | Description |
|------|-------------|
| `streamlit_app.py` | Main application logic and Streamlit UI |
| `requirements.txt` | Python dependencies to install |
| `best.pt` | YOLO model weights (rename your trained weights to this file name) |

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate           # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
streamlit run streamlit_app.py
The app will launch at http://localhost:8501.
Upload an image or toggle the webcam mode, then click "Run detection" to see results in real time!

## Configuration Notes
Use custom weights
Rename your model or set an environment variable:
export LOCAL_MODEL=my_model.pt
PyTorch on Apple Silicon
If torch installs slowly, try:
```bash
pip install 'torch==2.4.*' --extra-index-url https://download.pytorch.org/whl/cpu
```
