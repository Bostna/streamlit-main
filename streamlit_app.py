import os
import json
import base64
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# =========================================================
# Config (local only)
# =========================================================
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "yolo_litter.pt")
CACHED_PATH = LOCAL_MODEL  # kept for compatibility

# =========================================================
# Optional logo (safe if missing)
# =========================================================
def _try_render_logo():
    logo_path = "logo/when_ai_sees_litter_logo.png"
    try:
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"""
            <style>
            .top-right {{ position:absolute; top:10px; right:10px; z-index:9999; }}
            </style>
            <div class="top-right"><img src="data:image/png;base64,{b64}" width="120"></div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

_try_render_logo()

# =========================================================
# Model file management (local only)
# =========================================================
def _ensure_model_path() -> str:
    if not os.path.exists(LOCAL_MODEL):
        st.error(
            f"Model file '{LOCAL_MODEL}' not found.\n"
            "Put yolo_litter.pt next to this file or set LOCAL_MODEL to the correct path."
        )
        st.stop()
    return LOCAL_MODEL

@st.cache_resource(show_spinner=True)
def load_model(cache_key: float = 0.0):
    path = _ensure_model_path()
    return YOLO(path)

def get_class_names(model: YOLO):
    names = getattr(getattr(model, "model", model), "names", None)
    if isinstance(names, dict):
        # sort by index to make a list
        return [names[i] for i in range(len(names))]
    if isinstance(names, list):
        return names
    # fallback to TACO-20 ordering if names are unavailable
    return [
        "Cigarette","Plastic film","Clear plastic bottle","Other plastic",
        "Other plastic wrapper","Drink can","Plastic bottle cap","Plastic straw",
        "Broken glass","Styrofoam piece","Glass bottle","Disposable plastic cup",
        "Pop tab","Other carton","Paper","Food wrapper","Metal bottle cap",
        "Cardboard","Paper cup","Plastic utensils"
    ]

# =========================================================
# Shibuya category mapping (simple rules)
# =========================================================
def shibuya_category_and_note(name: str):
    cat, note = "Burnable (default) or check ward", "If plastic is clean → recyclable; if not washable → burnable."
    paper_like = {"Paper", "Cardboard", "Other carton"}
    plastics = {
        "Plastic film","Other plastic","Other plastic wrapper","Plastic bottle cap",
        "Plastic straw","Disposable plastic cup","Plastic utensils","Food wrapper","Styrofoam piece"
    }
    if name in paper_like:
        if name == "Cardboard":
            cat, note = "Recyclable — Cardboard", "Bundle by type; no bags."
        else:
            cat, note = "Recyclable — Paper", "Bundle by type; if soiled → burnable."
    elif name == "Clear plastic bottle":
        cat, note = "Recyclable — PET bottle", "Remove cap+label, rinse; crush."
    elif name == "Drink can":
        cat, note = "Recyclable — Can", "Rinse; place in clear bag."
    elif name == "Glass bottle":
        cat, note = "Recyclable — Bottle", "Remove cap; rinse. Heavily soiled or broken → non-burnable."
    elif name in plastics:
        cat, note = "Recyclable — Plastics (if clean)", "Lightly rinse; non-washable → burnable."
    elif name in {"Metal bottle cap","Pop tab"}:
        cat, note = "Non-burnable (small metal)", "Clear or semi-clear bag."
    elif name == "Broken glass":
        cat, note = "Non-burnable (glass/ceramic)", "Wrap and mark 'キケン'."
    elif name == "Cigarette":
        cat, note = "Burnable", "Fully extinguish before disposal."
    return cat, note

# =========================================================
# Inference helpers (image)
# =========================================================
def pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))

def run_inference_rgb(rgb: np.ndarray, conf: float, iou: float, imgsz: int):
    model = load_model()
    names = get_class_names(model)

    results = model.predict(rgb, conf=conf, iou=iou, imgsz=imgsz, max_det=300, verbose=False)
    pred = results[0]

    dets = []
    if pred.boxes is not None and len(pred.boxes) > 0:
        boxes = pred.boxes.xyxy.cpu().numpy()
        scores = pred.boxes.conf.cpu().numpy()
        clsi   = pred.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            c = int(clsi[i])
            name = names[c] if 0 <= c < len(names) else str(c)
            s = float(scores[i])
            sh_cat, note = shibuya_category_and_note(name)
            dets.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_id": c, "class_name": name, "score": s,
                "shibuya_type": sh_cat, "note": note
            })

    anno_bgr = pred.plot()
    anno_rgb = anno_bgr[:, :, ::-1]
    vis = Image.fromarray(anno_rgb)

    counts = {}
    for d in dets:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1

    task = getattr(getattr(model, "model", model), "task", "unknown")
    return dets, counts, vis, names, task

# =========================================================
# Video processing (CPU-friendly sampling)
# =========================================================
def process_video(file_bytes: bytes, conf: float, iou: float, imgsz: int,
                  frame_step: int = 3, max_frames: int = 600):
    import cv2

    in_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    in_file.write(file_bytes)
    in_file.flush(); in_file.close()

    cap = cv2.VideoCapture(in_file.name)
    if not cap.isOpened():
        st.error("Could not read the uploaded video.")
        return None, {}, {}, 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps   = max(1.0, cap.get(cv2.CAP_PROP_FPS) or 24.0)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_HEIGHT))

    fps_out = max(1.0, fps / max(1, frame_step))
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = cv2.VideoWriter(out_file.name, fourcc, fps_out, (w, h))

    model = load_model()
    names = get_class_names(model)

    class_counts = {}
    shibuya_counts = {}
    frame_idx = 0
    processed = 0
    target_steps = min(max_frames, total // max(1, frame_step) if total else max_frames)
    prog = st.progress(0.0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, frame_step) != 0:
            frame_idx += 1
            continue

        results = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, max_det=300, verbose=False)
        pred = results[0]

        anno = pred.plot()  # BGR
        if anno.shape[1] != w or anno.shape[0] != h:
            anno = cv2.resize(anno, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(anno)

        if pred.boxes is not None and len(pred.boxes) > 0:
            clsi  = pred.boxes.cls.cpu().numpy().astype(int)
            for c in clsi:
                name = names[c] if 0 <= c < len(names) else str(c)
                class_counts[name] = class_counts.get(name, 0) + 1
                sh_cat, _ = shibuya_category_and_note(name)
                shibuya_counts[sh_cat] = shibuya_counts.get(sh_cat, 0) + 1

        processed += 1
        frame_idx += 1
        if processed >= max_frames:
            break
        if target_steps:
            prog.progress(min(1.0, processed / target_steps))

    cap.release(); writer.release()
    prog.progress(1.0)
    return out_file.name, class_counts, shibuya_counts, processed

# =========================================================
# UI
# =========================================================
st.title("When AI Sees Litter")
st.write("Detect items and map to Shibuya garbage type ♻️")
st.caption("Runs inference with your local YOLO weights: yolo_litter.pt")

# Controls
col1, col2, col3 = st.columns(3)
conf  = col1.slider("Confidence", 0.01, 0.90, 0.15, 0.01)
iou   = col2.slider("IoU",        0.10, 0.90, 0.45, 0.01)
imgsz = col3.select_slider("Image size", options=[320, 480, 640, 800, 960, 1024], value=960)

mode = st.selectbox("Input type", ("Capture Photo (Webcam)", "Upload Image", "Upload Video"))

if st.button("🔁 Load model"):
    _ = load_model()
    st.success("Model loaded.")

# Webcam
if mode == "Capture Photo (Webcam)":
    shot = st.camera_input("Take a picture", key="cam_photo")  # unique key avoids duplicate-ID errors
    if shot is not None:
        img = Image.open(shot).convert("RGB")
        st.image(img, caption="Captured photo", use_container_width=True)
        if st.button("Run detection on photo"):
            rgb = pil_to_rgb(img)
            dets, counts, vis, names, task = run_inference_rgb(rgb, conf, iou, imgsz)
            st.caption(f"Model task: {task} • Classes: {len(names)}")
            st.subheader("Detections")
            st.image(vis, use_container_width=True)
            if dets:
                st.dataframe(pd.DataFrame(dets))
                by_sh = pd.Series([d["shibuya_type"] for d in dets]).value_counts()
                st.subheader("Counts by Shibuya type")
                st.bar_chart(by_sh)
            else:
                st.info("No detections. Try lower confidence, higher image size, or a clearer object.")

# Image
elif mode == "Upload Image":
    up = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if up is not None:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)
        if st.button("Run detection on image"):
            rgb = pil_to_rgb(img)
            dets, counts, vis, names, task = run_inference_rgb(rgb, conf, iou, imgsz)
            st.caption(f"Model task: {task} • Classes: {len(names)}")
            st.subheader("Detections")
            st.image(vis, use_container_width=True)
            if dets:
                st.dataframe(pd.DataFrame(dets))
                by_sh = pd.Series([d["shibuya_type"] for d in dets]).value_counts()
                st.subheader("Counts by Shibuya type")
                st.bar_chart(by_sh)
            else:
                st.info("No detections. Try lower confidence, higher image size, or a clearer object.")

# Video
else:
    vid = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])
    step = st.slider("Process every Nth frame", 1, 12, 3)
    cap_limit = st.slider("Max processed frames", 100, 2500, 600, step=50)
    if vid is not None:
        st.video(vid)
        if st.button("Run detection on video"):
            with st.spinner("Processing video on CPU…"):
                out_path, class_counts, shibuya_counts, processed = process_video(
                    vid.read(), conf=conf, iou=iou, imgsz=imgsz, frame_step=step, max_frames=cap_limit
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
