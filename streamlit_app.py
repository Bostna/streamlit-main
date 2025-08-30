import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# =========================================================
# Config
# Options:
#   1) Local file  -> put yolo_litter.pt next to this file or set LOCAL_MODEL
#   2) GitHub RAW  -> set USE_GITHUB=1 and provide GITHUB_URL
# Env examples:
#   USE_GITHUB=1
#   GITHUB_URL=https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/to/yolo_litter.pt
# =========================================================
USE_GITHUB   = os.getenv("USE_GITHUB", "0") == "1"
GITHUB_URL   = os.getenv("GITHUB_URL", "")
LOCAL_MODEL  = os.getenv("LOCAL_MODEL", "yolo_litter.pt")
CACHED_DIR   = "/tmp/models"
CACHED_PATH  = os.path.join(CACHED_DIR, "yolo_litter.pt")

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
    """Return a local path to weights. Download from GitHub if needed."""
    os.makedirs(CACHED_DIR, exist_ok=True)

    if USE_GITHUB:
        if not GITHUB_URL:
            st.error("USE_GITHUB=1 but GITHUB_URL is empty. Provide a raw URL to yolo_litter.pt.")
            st.stop()
        if not os.path.exists(CACHED_PATH):
            try:
                import requests
                with st.spinner("Downloading model from GitHub..."):
                    resp = requests.get(GITHUB_URL, timeout=60)
                    resp.raise_for_status()
                    with open(CACHED_PATH, "wb") as f:
                        f.write(resp.content)
                st.success("Downloaded model from GitHub.")
            except Exception as e:
                st.error(f"Failed to download from GitHub\n{e}")
                st.stop()
        return CACHED_PATH

    # Local fallback
    if not os.path.exists(LOCAL_MODEL):
        st.error(f"Model file '{LOCAL_MODEL}' not found. Put yolo_litter.pt next to this file or set LOCAL_MODEL")
        st.stop()
    return LOCAL_MODEL

@st.cache_resource(show_spinner=True)
def load_model():
    path = _ensure_model_path()
    return YOLO(path)

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return arr[:, :, ::-1]

def _class_name_lookup(model, idx: int) -> str:
    try:
        names = getattr(model.model, "names", None) or getattr(model, "names", None)
        if isinstance(names, dict):
            return names.get(int(idx), str(int(idx)))
        if isinstance(names, list) and 0 <= int(idx) < len(names):
            return names[int(idx)]
    except Exception:
        pass
    if 0 <= int(idx) < len(CLASS_NAMES):
        return CLASS_NAMES[int(idx)]
    return str(int(idx))

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
    shot = st.camera_input("Take a photo", key="cam_one")
    if shot:
        image = Image.open(shot).convert("RGB")

if st.button("Load model"):
    model = load_model()
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
                name = _class_name_lookup(model, c)
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
