import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# =========================================================
# Config
# - Step 1: keep USE_GCS=0 and put best.pt next to this file.
# - Step 2: set USE_GCS=1 and provide GCS_BUCKET + GCS_BLOB.
# =========================================================
USE_GCS     = os.getenv("USE_GCS", "0") == "1"
GCS_BUCKET  = os.getenv("GCS_BUCKET", "tacov2")  # e.g. taco_2025_08_16
GCS_BLOB    = os.getenv("GCS_BLOB", "taco_env/TACO/derived/_artifacts/cv/cv_20250827_032326_fold4_best_20250827_143649.pt")    # e.g. taco_env/TACO/derived/_artifacts/weights/best.pt
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "best.pt")
CACHED_PATH = "/tmp/models/best.pt"

CLASS_NAMES = [
    "Cigarette","Plastic film","Clear plastic bottle","Other plastic",
    "Other plastic wrapper","Drink can","Plastic bottle cap","Plastic straw",
    "Broken glass","Styrofoam piece","Glass bottle","Disposable plastic cup",
    "Pop tab","Other carton","Paper","Food wrapper","Metal bottle cap",
    "Cardboard","Paper cup","Plastic utensils"
]

# =========================================================
# Helpers
# =========================================================
def _ensure_model_path() -> str:
    """Get a local path to weights. If USE_GCS=1, download once to /tmp/models."""
    if USE_GCS:
        os.makedirs("/tmp/models", exist_ok=True)
        if not os.path.exists(CACHED_PATH):
            try:
                from google.cloud import storage
                client = storage.Client()  # ADC or SA key via env var
                blob = client.bucket(GCS_BUCKET).blob(GCS_BLOB)
                blob.download_to_filename(CACHED_PATH)
            except Exception as e:
                st.error(f"Failed to download model from gs://{GCS_BUCKET}/{GCS_BLOB}\n{e}")
                st.stop()
        return CACHED_PATH
    else:
        if not os.path.exists(LOCAL_MODEL):
            st.error(
                f"Model file '{LOCAL_MODEL}' not found.\n"
                "• For Step 1: put best.pt next to this file or set LOCAL_MODEL\n"
                "• For Step 2: set USE_GCS=1 with GCS_BUCKET + GCS_BLOB"
            )
            st.stop()
        return LOCAL_MODEL

@st.cache_resource(show_spinner=True)
def load_model():
    path = _ensure_model_path()
    # Optional: let YOLO decide device automatically
    return YOLO(path)

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return arr[:, :, ::-1]

def draw_boxes(bgr, dets):
    import cv2
    out = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
        label = f'{d["class_name"]} {d["score"]:.2f}'
        y = max(y1 - 7, 7)
        cv2.putText(out, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return Image.fromarray(out[:, :, ::-1])

# =========================================================
# UI
# =========================================================
st.title("TACO Detect")

col1, col2 = st.columns(2)
conf = col1.slider("Confidence", 0.05, 0.95, 0.25, 0.01)
iou  = col2.slider("IoU", 0.10, 0.90, 0.45, 0.01)

src = st.radio("Input source", ["Upload image", "Webcam"], horizontal=True)
image = None
if src == "Upload image":
    up = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if up:
        image = Image.open(up).convert("RGB")
else:
    shot = st.camera_input("Take a photo")
    if shot:
        image = Image.open(shot).convert("RGB")

if st.button("Load model"):
    _ = load_model()
    st.success("Model ready.")

if image is not None:
    st.image(image, caption="Input", use_container_width=True)
    if st.button("Run detection"):
        model = load_model()
        bgr = pil_to_bgr(image)
        results = model.predict(bgr, conf=conf, iou=iou, imgsz=640, verbose=False)
        pred = results[0]

        if pred.boxes is None or len(pred.boxes) == 0:
            st.info("No detections")
        else:
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            clsi   = pred.boxes.cls.cpu().numpy().astype(int)

            dets, counts = [], {}
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                c = int(clsi[i])
                name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c)
                s = float(scores[i])
                dets.append({"xyxy":[x1,y1,x2,y2], "class_id":c, "class_name":name, "score":s})
                counts[name] = counts.get(name, 0) + 1

            vis_img = draw_boxes(bgr, dets)
            st.subheader("Detections")
            st.image(vis_img, use_container_width=True)

            st.subheader("Raw detections")
            st.dataframe(pd.DataFrame(dets))

            if counts:
                st.subheader("Counts")
                st.bar_chart(pd.Series(counts).sort_values(ascending=False))
