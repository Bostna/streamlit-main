import os
import io
import base64
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================
# Set these in Streamlit Cloud → Settings → Advanced → Environment variables:
# USE_GCS=1
# GCS_BUCKET=ptfile
# GCS_BLOB=taco_env_TACO_derived__artifacts_cv_cv_20250827_032326_fold0_best_20250827_130937.pt
USE_GCS     = os.getenv("USE_GCS", "0") == "1"
GCS_BUCKET  = os.getenv("GCS_BUCKET", "")
GCS_BLOB    = os.getenv("GCS_BLOB", "")
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
# OPTIONAL LOGO
# =========================================================
def _try_render_logo():
    logo_path = "logo/when_ai_sees_litter_logo.png"
    try:
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"""
            <style>
            .top-right {{
                position: absolute; top: 10px; right: 10px; z-index: 9999;
            }}
            </style>
            <div class="top-right">
                <img src="data:image/png;base64,{b64}" width="120">
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        pass

_try_render_logo()

# =========================================================
# MODEL LOADING (local or GCS)
# =========================================================
def _ensure_model_path() -> str:
    """Return local path to YOLO weights. If USE_GCS=1, download once to /tmp."""
    if USE_GCS:
        os.makedirs("/tmp/models", exist_ok=True)
        if not os.path.exists(CACHED_PATH):
            try:
                from google.cloud import storage
                if "gcp_service_account" in st.secrets:
                    from google.oauth2 import service_account
                    creds = service_account.Credentials.from_service_account_info(
                        st.secrets["gcp_service_account"]
                    )
                    client = storage.Client(credentials=creds)
                else:
                    client = storage.Client()
                client.bucket(GCS_BUCKET).blob(GCS_BLOB).download_to_filename(CACHED_PATH)
            except Exception as e:
                st.error(
                    "Failed to download model from "
                    f"gs://{GCS_BUCKET}/{GCS_BLOB}\n\n{e}"
                )
                st.stop()
        return CACHED_PATH
    else:
        if not os.path.exists(LOCAL_MODEL):
            st.error(
                f"Model file '{LOCAL_MODEL}' not found.\n"
                "• Local run: put best.pt next to this file or set LOCAL_MODEL\n"
                "• Cloud (GCS): set USE_GCS=1 and provide GCS_BUCKET + GCS_BLOB"
            )
            st.stop()
        return LOCAL_MODEL

@st.cache_resource(show_spinner=True)
def load_model():
    path = _ensure_model_path()
    return YOLO(path)

# =========================================================
# SHIBUYA CATEGORY MAPPER
# =========================================================
def shibuya_category_and_note(taco_name: str):
    """Map TACO class → Shibuya garbage category + disposal note.
       Sources: Shibuya City 'Garbage and Recycling' page (EN)."""
    # Default
    cat = "—"
    note = "Check item locally."

    # Recyclables (paper)
    paper_like = {"Paper", "Cardboard", "Other carton"}
    # Plastics (if clean)
    plastic_like = {
        "Plastic film","Other plastic","Other plastic wrapper","Plastic bottle cap",
        "Plastic straw","Disposable plastic cup","Plastic utensils","Food wrapper","Styrofoam piece"
    }

    if taco_name in paper_like:
        if taco_name == "Cardboard":
            cat = "Recyclable resource — Cardboard"
            note = "Bundle by type; no bags."
        elif taco_name in {"Paper", "Other carton"}:
            cat = "Recyclable resource — Paper"
            note = "Bundle newspapers/magazines/boxes by type; no bags. If soiled → burnable."

    elif taco_name == "Clear plastic bottle":
        cat = "Recyclable resource — PET bottle"
        note = "Remove cap+label, rinse, crush. Oil/cosmetic/paint PET bottles not accepted; heavily soiled PET → burnable."

    elif taco_name == "Drink can":
        cat = "Recyclable resource — Can"
        note = "Rinse; place in clear bag."

    elif taco_name == "Glass bottle":
        cat = "Recyclable resource — Bottle"
        note = "Remove cap; rinse. Heavily soiled/damaged → non-burnable."

    elif taco_name in plastic_like:
        cat = "Recyclable resource — Plastics (if clean)"
        note = "Lightly rinse plastics and bag transparently; non-washable/dirty plastics → burnable."

    elif taco_name in {"Metal bottle cap", "Pop tab"}:
        cat = "Non-burnable (small metal)"
        note = "Put in clear/semiclear bag."

    elif taco_name == "Broken glass":
        cat = "Non-burnable (glass/ceramic)"
        note = "Wrap sharp edges; mark 'キケン' (danger)."

    elif taco_name == "Cigarette":
        cat = "Burnable (regular)"
        note = "Fully extinguish before disposal."

    elif taco_name == "Paper cup":
        cat = "Burnable (paper with lining)"
        note = "Paper cups usually burnable unless accepted by special group collection."

    else:
        cat = "Burnable (default) or check ward"
        note = "If plastic and clean → plastics (recyclable); if not washable → burnable."

    return cat, note

# =========================================================
# INFERENCE HELPERS
# =========================================================
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return arr[:, :, ::-1]  # RGB→BGR

def draw_boxes_on_bgr(bgr: np.ndarray, dets: list) -> Image.Image:
    import cv2
    out = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{d["class_name"]} {d["score"]:.2f}'
        y = max(y1 - 7, 7)
        cv2.putText(out, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return Image.fromarray(out[:, :, ::-1])  # Back to RGB for Streamlit

def run_yolo_on_bgr(bgr: np.ndarray, conf: float, iou: float):
    model = load_model()
    results = model.predict(bgr, conf=conf, iou=iou, imgsz=640, verbose=False)
    pred = results[0]

    dets = []
    counts = {}
    if pred.boxes is not None and len(pred.boxes) > 0:
        boxes = pred.boxes.xyxy.cpu().numpy()
        scores = pred.boxes.conf.cpu().numpy()
        clsi   = pred.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            c = int(clsi[i])
            name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c)
            s = float(scores[i])
            shibuya_cat, note = shibuya_category_and_note(name)
            dets.append({
                "xyxy":[x1,y1,x2,y2],
                "class_id": c,
                "class_name": name,
                "score": s,
                "shibuya_type": shibuya_cat,
                "note": note
            })
            counts[name] = counts.get(name, 0) + 1

    vis = draw_boxes_on_bgr(bgr, dets)
    return dets, counts, vis

def process_video(file_bytes: bytes, conf: float, iou: float, frame_step: int = 3, max_frames: int = 600):
    """Annotate a video; sample every `frame_step` frames for speed."""
    import cv2

    # Save upload to temp for OpenCV
    in_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    in_file.write(file_bytes)
    in_file.flush(); in_file.close()

    cap = cv2.VideoCapture(in_file.name)
    if not cap.isOpened():
        st.error("Could not read the uploaded video.")
        return None, {}, {}, 0

    fps   = max(1.0, cap.get(cv2.CAP_PROP_FPS))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = max(1.0, fps / max(1, frame_step))
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = cv2.VideoWriter(out_file.name, fourcc, fps_out, (w, h))

    model = load_model()
    class_counts = {}
    shibuya_counts = {}
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, frame_step) != 0:
            frame_idx += 1
            continue

        results = model.predict(frame, conf=conf, iou=iou, imgsz=640, verbose=False)
        pred = results[0]

        if pred.boxes is not None and len(pred.boxes) > 0:
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            clsi   = pred.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i].tolist())
                c = int(clsi[i])
                name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c)
                s = float(scores[i])

                # counts
                class_counts[name] = class_counts.get(name, 0) + 1
                shibuya_cat, _ = shibuya_category_and_note(name)
                shibuya_counts[shibuya_cat] = shibuya_counts.get(shibuya_cat, 0) + 1

                # draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                label = f"{name} {s:.2f}"
                ytxt = max(y1 - 7, 7)
                cv2.putText(frame, label, (x1, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        writer.write(frame)
        processed += 1
        frame_idx += 1
        if processed >= max_frames:
            break

    cap.release(); writer.release()
    return out_file.name, class_counts, shibuya_counts, processed

# =========================================================
# UI
# =========================================================
st.title("When AI Sees Litter")
st.write("Identify items and show the Shibuya garbage type ♻️")
st.caption("Model runs locally with weights from GCS or a local .pt file.")

col1, col2 = st.columns(2)
conf = col1.slider("Confidence", 0.05, 0.95, 0.25, 0.01)
iou  = col2.slider("IoU",        0.10, 0.90, 0.45, 0.01)

mode = st.selectbox("Input type", ("Capture Photo (Webcam)", "Upload Image", "Upload Video"))

if st.button("🔁 Load model"):
    _ = load_model()
    st.success("Model loaded.")

# Webcam
if mode == "Capture Photo (Webcam)":
    shot = st.camera_input("Take a picture")
    if shot is not None:
        img = Image.open(shot).convert("RGB")
        st.image(img, caption="Captured photo", use_container_width=True)
        if st.button("Run detection on photo"):
            bgr = np.array(img)[:, :, ::-1]
            dets, counts, vis = run_yolo_on_bgr(bgr, conf, iou)
            st.subheader("Detections")
            st.image(vis, use_container_width=True)
            if dets:
                st.dataframe(pd.DataFrame(dets))
            if counts:
                st.subheader("Counts by class")
                st.bar_chart(pd.Series(counts).sort_values(ascending=False))

# Image
elif mode == "Upload Image":
    up = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if up is not None:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)
        if st.button("Run detection on image"):
            bgr = np.array(img)[:, :, ::-1]
            dets, counts, vis = run_yolo_on_bgr(bgr, conf, iou)
            st.subheader("Detections")
            st.image(vis, use_container_width=True)
            if dets:
                st.dataframe(pd.DataFrame(dets))
                # Aggregate by Shibuya type
                by_shibuya = pd.Series([d["shibuya_type"] for d in dets]).value_counts()
                st.subheader("Counts by Shibuya garbage type")
                st.bar_chart(by_shibuya)
            if counts:
                st.subheader("Counts by class")
                st.bar_chart(pd.Series(counts).sort_values(ascending=False))

# Video
else:
    vid = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])
    step = st.slider("Process every Nth frame", 1, 10, 3)
    cap_limit = st.slider("Max processed frames", 100, 2000, 600, step=50)
    if vid is not None:
        st.video(vid)
        if st.button("Run detection on video"):
            with st.spinner("Processing video…"):
                out_path, class_counts, shibuya_counts, processed = process_video(
                    vid.read(), conf=conf, iou=iou, frame_step=step, max_frames=cap_limit
                )
            if out_path:
                st.success(f"Processed {processed} sampled frames.")
                st.subheader("Annotated Video")
                st.video(out_path)
                with open(out_path, "rb") as f:
                    st.download_button("Download annotated MP4", f, file_name="annotated.mp4", mime="video/mp4")
                if shibuya_counts:
                    st.subheader("Counts by Shibuya garbage type (sampled)")
                    st.bar_chart(pd.Series(shibuya_counts).sort_values(ascending=False))
                if class_counts:
                    st.subheader("Counts by class (sampled)")
                    st.bar_chart(pd.Series(class_counts).sort_values(ascending=False))
            else:
                st.error("Video processing failed.")


