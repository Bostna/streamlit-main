import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="TACO Detect", page_icon="♻️", layout="wide")

# -------------------------
# Config
# -------------------------
MODEL_URL     = os.getenv("MODEL_URL", "https://raw.githubusercontent.com/Bellzum/streamlit-main/main/new_taco1.pt")
LOCAL_MODEL   = os.getenv("LOCAL_MODEL", "best.pt")
CACHED_PATH   = "/tmp/models/best.pt"
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))

# Fallback names if checkpoint has none
CLASS_NAMES   = ["Clear plastic bottle", "Drink can", "Plastic bottle cap"]
IMGSZ_OPTIONS = [320, 416, 512, 640, 800, 960, 1280]

# -------------------------
# Shibuya disposal guidance + facts
# -------------------------
SHIBUYA_GUIDE_URL = "https://www.city.shibuya.tokyo.jp/contents/living-in-shibuya/en/daily/garbage.html"

GUIDE = {
    "Clear plastic bottle": {
        "title": "PET bottle (resource)",
        "emoji": "🧴",
        "steps": [
            "Remove the cap and label.",
            "Rinse the bottle.",
            "Crush it flat.",
            "Put PET bottles in a transparent bag for PET.",
            "Put caps and labels with Plastics."
        ],
        "facts": [
            {
                # PET bottles are recycled into new bottles, sheets, and fiber for clothing like uniforms and bags
                "text": "In Japan, used PET bottles become new bottles, sheet products, and fibers for clothing such as uniforms and bags.",
                "url": "https://www.petbottle-rec.gr.jp/qanda/sec7.html"
            },
            {
                # Bottle-to-bottle is common in Japan
                "text": "Bottle-to-bottle recycling is widely implemented in Japan.",
                "url": "https://www.suntory.com/csr/story/003/"
            }
        ],
        "link": SHIBUYA_GUIDE_URL,
    },
    "Drink can": {
        "title": "Aluminum or steel can (resource)",
        "emoji": "🥫",
        "steps": [
            "Rinse the can.",
            "Put cans in a transparent bag for cans."
        ],
        "facts": [],
        "link": SHIBUYA_GUIDE_URL,
    },
    "Plastic bottle cap": {
        "title": "Plastic bottle cap (plastics)",
        "emoji": "🔘",
        "steps": [
            "Remove from the bottle.",
            "If dirty, rinse quickly.",
            "Put caps with Plastics in a transparent bag.",
            "Do not put caps with the PET-bottle bag."
        ],
        "facts": [
            {
                # Shibuya note: caps are plastic, not PET
                "text": "Shibuya treats caps and labels as Plastics, separate from PET bottles.",
                "url": "https://files.city.shibuya.tokyo.jp/assets/12995aba8b194961be709ba879857f70/0cdf099fdfe8456fbac12bb5ad7927e4/assets_kusei_ShibuyaCityNews2206_e.pdf"
            },
            {
                # Cap-to-cap horizontal recycling pilot in Japan
                "text": "Japan is piloting horizontal recycling for plastic bottle caps (cap-to-cap).",
                "url": "https://www.sojitz.com/en/news/article/topics-20230112_02.html"
            }
        ],
        "link": SHIBUYA_GUIDE_URL,
    },
}

def show_shibuya_guidance(label: str, count: int = 0):
    info = GUIDE.get(label)
    if not info:
        return
    st.markdown(f"### {info['emoji']} Shibuya disposal: {info['title']}")
    if count:
        st.caption(f"Detected: {count}")
    for step in info["steps"]:
        st.write(f"• {step}")

    # Short educational card
    facts = info.get("facts", [])
    if facts:
        with st.container(border=True):
            st.markdown("**Did you know?**")
            for fact in facts:
                st.write(f"• {fact['text']}")
            # Combine into a single row of links
            cols = st.columns(len(facts))
            for i, fact in enumerate(facts):
                with cols[i]:
                    st.link_button("Learn more", fact["url"])

    # Always offer the official Shibuya page
    try:
        st.link_button("Official guidance", info["link"])
    except Exception:
        st.markdown(f"[Official guidance]({info['link']})")

# -------------------------
# Helpers
# -------------------------
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
                "If this is a private repo or rate limit issue, make the file public or commit it to this repo."
            )
            st.stop()

def _ensure_model_path() -> str:
    if MODEL_URL.strip().startswith("http"):
        if not os.path.exists(CACHED_PATH):
            _download_file(MODEL_URL.strip(), CACHED_PATH)
        return CACHED_PATH
    if not os.path.exists(LOCAL_MODEL):
        st.error(f"Model file '{LOCAL_MODEL}' not found. Provide MODEL_URL or place best.pt next to this file.")
        st.stop()
    return LOCAL_MODEL

def _cache_key_for(path: str) -> str:
    try:
        return f"{path}:{os.path.getmtime(path)}:{os.path.getsize(path)}"
    except Exception:
        return path

@st.cache_resource(show_spinner=True)
def _load_model_cached(path: str, key: str):
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

# -------------------------
# UI
# -------------------------
st.title("TACO Detect")

with st.expander("Model source"):
    st.write(f"LOCAL_MODEL: {LOCAL_MODEL}")
    st.write(f"MODEL_URL: {MODEL_URL or '(empty)'}")
    if os.path.exists(CACHED_PATH):
        st.write(f"Cached path: {CACHED_PATH}  size: {os.path.getsize(CACHED_PATH)/1e6:.2f} MB")

col1, col2, col3 = st.columns(3)
conf = col1.slider("Base confidence", 0.05, 0.95, 0.25, 0.01, help="Passed into the model. We also apply per-class thresholds below.")
iou  = col2.slider("IoU", 0.10, 0.90, 0.45, 0.01)
default_imgsz = _closest_size(DEFAULT_IMGSZ, IMGSZ_OPTIONS)
imgsz = col3.select_slider("Inference image size", options=IMGSZ_OPTIONS, value=default_imgsz)

c1, c2, c3, c4 = st.columns(4)
bottle_min = c1.slider("Min conf: Bottle", 0.0, 1.0, 0.60, 0.01)
can_min    = c2.slider("Min conf: Can",    0.0, 1.0, 0.55, 0.01)
cap_min    = c3.slider("Min conf: Cap",    0.0, 1.0, 0.65, 0.01)
min_area_pct = c4.slider("Min box area (%)", 0.0, 5.0, 0.3, 0.1, help="Ignore tiny boxes by area percent of image")

tta = st.toggle("Test-time augmentation (slower, sometimes fewer false positives)", value=False)

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
    elif isinstance(names_from_model, list):
        st.info(f"Checkpoint labels: {names_from_model}")
    else:
        st.info(f"Using fallback CLASS_NAMES: {CLASS_NAMES}")
    st.success("Model ready.")

# -------------------------
# Inference
# -------------------------
if image is not None:
    st.image(image, caption="Input", use_container_width=True)
    if st.button("Run detection"):
        model = load_model()
        bgr = pil_to_bgr(image)
        results = model.predict(bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False, augment=tta)
        pred = results[0]

        if pred.boxes is None or len(pred.boxes) == 0:
            st.info("No detections")
        else:
            boxes  = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            clsi   = pred.boxes.cls.cpu().numpy().astype(int)

            names_map = _get_names_map(pred, model)

            # Per-class thresholds map
            per_class_min = {
                "Clear plastic bottle": bottle_min,
                "Drink can": can_min,
                "Plastic bottle cap": cap_min
            }

            H, W = bgr.shape[:2]
            min_area = (min_area_pct / 100.0) * (H * W)

            dets, counts = [], {}
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
                area = w * h

                c = int(clsi[i])
                if isinstance(names_map, dict):
                    name = names_map.get(c, str(c))
                else:
                    name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c)
                s = float(scores[i])

                # Apply extra filters
                thr = per_class_min.get(name, conf)
                if s < thr:
                    continue
                if area < min_area:
                    continue

                dets.append({"xyxy": [x1, y1, x2, y2], "class_id": c, "class_name": name, "score": s})
                counts[name] = counts.get(name, 0) + 1

            if not dets:
                st.info("All detections were filtered by thresholds. Try lowering per-class thresholds or min box area.")
            else:
                vis_img = draw_boxes(bgr, dets)
                st.subheader("Detections")
                st.image(vis_img, use_container_width=True)

                st.subheader("Raw detections")
                st.dataframe(pd.DataFrame(dets))

                if counts:
                    st.subheader("Counts")
                    st.bar_chart(pd.Series(counts).sort_values(ascending=False))

                # -------------------------
                # Shibuya instructions + facts for detected target classes
                # -------------------------
                detected_labels = sorted({d["class_name"] for d in dets})
                guide_labels = [lbl for lbl in detected_labels if lbl in GUIDE]

                if guide_labels:
                    st.subheader("Disposal instructions for Shibuya")
                    for lbl in guide_labels:
                        show_shibuya_guidance(lbl, counts.get(lbl, 0))
                else:
                    st.caption("No Shibuya guidance to show for these detections.")
