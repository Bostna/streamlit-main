import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="TACO Detect", page_icon="♻️", layout="wide")

# =========================================================
# Config
#   Option A: put best.pt next to this file and leave MODEL_URL empty
#   Option B: set MODEL_URL to a direct GitHub Raw URL to best.pt
#   You can set IMGSZ via environment, e.g. IMGSZ=640
# =========================================================
MODEL_URL    = os.getenv("MODEL_URL", "https://raw.githubusercontent.com/Bellzum/streamlit-main/main/new_taco1.pt")
LOCAL_MODEL  = os.getenv("LOCAL_MODEL", "best.pt")
CACHED_PATH  = "/tmp/models/best.pt"
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))  # only affects UI default

# Exactly 3 classes as a fallback
CLASS_NAMES = ["Clear plastic bottle", "Drink can", "Styrofoam piece"]

# Allowed image sizes for the select slider
IMGSZ_OPTIONS = [320, 416, 512, 640, 800, 960, 1280]

# =========================================================
# Helpers
# =========================================================
def _download_file(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return
    except Exception as e_req:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
        except Exception as e_url:
            st.error(
                f"Failed to download model from URL:\n{url}\n\n"
                f"requests error: {e_req}\nurllib error: {e_url}\n\n"
                "If this is a private repo or a rate limit issue, either make the file public, "
                "use a GitHub Release asset, or commit the weight into this app repository."
            )
            st.stop()

def _ensure_model_path() -> str:
    """Return a local path to the weights. Prefer MODEL_URL if set, otherwise LOCAL_MODEL."""
    if MODEL_URL.strip().startswith("http"):
        if not os.path.exists(CACHED_PATH):
            _download_file(MODEL_URL.strip(), CACHED_PATH)
        return CACHED_PATH

    if not os.path.exists(LOCAL_MODEL):
        st.error(
            f"Model file '{LOCAL_MODEL}' not found.\n"
            "Provide MODEL_URL as a direct GitHub Raw link or place best.pt next to this file."
        )
        st.stop()
    return LOCAL_MODEL

def _cache_key_for(path: str) -> str:
    try:
        return f"{path}:{os.path.getmtime(path)}:{os.path.getsize(path)}"
    except Exception:
        return path

@st.cache_resource(show_spinner=True)
def _load_model_cached(path: str, key: str):
    # key is unused by the body but ensures cache refresh when file changes
    return YOLO(path)

def load_model():
    path = _ensure_model_path()
    return _load_model_cached(path, _cache_key_for(path))

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return arr[:, :, ::-1]

def draw_boxes(bgr, dets):
    import cv2
    out = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{d["class_name"]} {d["score"]:.2f}'
        y = max(y1 - 7, 7)
        cv2.putText(out, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return Image.fromarray(out[:, :, ::-1])

def _get_names_map(pred, model):
    # Prefer names stored in prediction or model if available
    names_map = None
    if hasattr(pred, "names") and isinstance(pred.names, dict):
        names_map = pred.names
    elif hasattr(model, "names") and isinstance(model.names, dict):
        names_map = model.names
    elif hasattr(model, "names") and isinstance(model.names, list):
        names_map = {i: n for i, n in enumerate(model.names)}
    return names_map

def _closest_size(target: int, options: list[int]) -> int:
    return min(options, key=lambda x: abs(x - target))

# =========================================================
# UI
# =========================================================
st.title("TACO Detect")

with st.expander("Model source"):
    st.write(f"LOCAL_MODEL: {LOCAL_MODEL}")
    st.write(f"MODEL_URL: {MODEL_URL or '(empty)'}")
    if os.path.exists(CACHED_PATH):
        st.write(f"Cached path: {CACHED_PATH}  size: {os.path.getsize(CACHED_PATH)/1e6:.2f} MB")

col1, col2, col3 = st.columns(3)
conf = col1.slider("Confidence", 0.05, 0.95, 0.25, 0.01)
iou  = col2.slider("IoU", 0.10, 0.90, 0.45, 0.01)

default_imgsz = _closest_size(DEFAULT_IMGSZ, IMGSZ_OPTIONS)
imgsz = col3.select_slider(
    "Inference image size",
    options=IMGSZ_OPTIONS,
    value=default_imgsz,
    help="Bigger size helps small objects but is slower"
)

src = st.radio("Input source", ["Upload image", "Webcam"], horizontal=True)
image = None
if src == "Upload image":
    up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if up:
        image = Image.open(up).convert("RGB")
else:
    shot = st.camera_input("Take a photo", key="cam1")
    if shot:
        image = Image.open(shot).convert("RGB")

if st.button("Load model"):
    m = load_model()
    names_from_model = getattr(m, "names", None)
    if isinstance(names_from_model, dict):
        st.info(f"Checkpoint labels: {list(names_from_model.values())}")
        if len(names_from_model) != 3:
            st.warning(f"Checkpoint reports {len(names_from_model)} classes. Expected 3.")
    elif isinstance(names_from_model, list):
        st.info(f"Checkpoint labels: {names_from_model}")
        if len(names_from_model) != 3:
            st.warning(f"Checkpoint reports {len(names_from_model)} classes. Expected 3.")
    else:
        st.info(f"Using fallback CLASS_NAMES: {CLASS_NAMES}")
    st.success("Model ready.")

if image is not None:
    st.image(image, caption="Input", use_container_width=True)
    if st.button("Run detection"):
        model = load_model()
        bgr = pil_to_bgr(image)
        results = model.predict(bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        pred = results[0]

        if pred.boxes is None or len(pred.boxes) == 0:
            st.info("No detections")
        else:
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            clsi   = pred.boxes.cls.cpu().numpy().astype(int)

            names_map = _get_names_map(pred, model)

            dets, counts = [], {}
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                c = int(clsi[i])
                if isinstance(names_map, dict):
                    name = names_map.get(c, str(c))
                else:
                    name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c)
                s = float(scores[i])
                dets.append({"xyxy": [x1, y1, x2, y2], "class_id": c, "class_name": name, "score": s})
                counts[name] = counts.get(name, 0) + 1

            vis_img = draw_boxes(bgr, dets)
            st.subheader("Detections")
            st.image(vis_img, use_container_width=True)

            st.subheader("Raw detections")
            st.dataframe(pd.DataFrame(dets))

            if counts:
                st.subheader("Counts")
                st.bar_chart(pd.Series(counts).sort_values(ascending=False))
