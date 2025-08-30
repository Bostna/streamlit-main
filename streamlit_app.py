import os
import json
import base64
import hashlib
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# =========================================================
# Config (prefer Secrets; fallback to env)
# =========================================================
def _cfg(key: str, default: str = "") -> str:
    return str(st.secrets.get(key, os.getenv(key, default)))

USE_GCS     = _cfg("USE_GCS", "0") == "1"
GCS_BUCKET  = _cfg("GCS_BUCKET", "")
GCS_BLOB    = _cfg("GCS_BLOB", "")
LOCAL_MODEL = _cfg("LOCAL_MODEL", "best.pt")

def _blob_cache_path() -> str:
    if not USE_GCS:
        return LOCAL_MODEL
    key = f"{GCS_BUCKET}/{GCS_BLOB}"
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return f"/tmp/models/{h}.pt"

CACHED_PATH = _blob_cache_path()

# Used to version the cached YOLO object in Streamlit
MODEL_SIG = f"{'gcs' if USE_GCS else 'local'}:{GCS_BUCKET}/{GCS_BLOB or LOCAL_MODEL}"

# =========================================================
# Optional logo (won’t crash if missing)
# =========================================================
def _try_render_logo():
    logo_path = "logo/when_ai_sees_litter_logo.png"
    try:
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"""
            <style>.top-right{{position:absolute;top:10px;right:10px;z-index:9999}}</style>
            <div class="top-right"><img src="data:image/png;base64,{b64}" width="120"></div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

_try_render_logo()

# =========================================================
# Service account & GCS client (no metadata server)
# =========================================================
def _read_sa_from_secrets():
    sa = st.secrets.get("gcp_service_account", None)
    if sa is None:
        return None
    if isinstance(sa, str):  # user pasted raw JSON as a string
        try:
            return json.loads(sa)
        except Exception:
            return None
    return dict(sa)

def _make_storage_client():
    from google.cloud import storage
    creds = None
    project = None

    sa = _read_sa_from_secrets()
    if sa:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_info(sa)
        project = sa.get("project_id")

    project = project or st.secrets.get("GOOGLE_CLOUD_PROJECT") \
              or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")

    # Pass project explicitly so it never tries GCE metadata server
    return storage.Client(credentials=creds, project=project)

def list_gcs(prefix: str):
    client = _make_storage_client()
    return [b.name for b in client.list_blobs(GCS_BUCKET, prefix=prefix)]

# =========================================================
# Model file management (GCS or local)
# =========================================================
def _download_model(force: bool = False) -> str:
    if not USE_GCS:
        return LOCAL_MODEL

    if not GCS_BUCKET or not GCS_BLOB:
        st.error("GCS is enabled but GCS_BUCKET or GCS_BLOB is empty.")
        st.stop()

    os.makedirs(os.path.dirname(CACHED_PATH), exist_ok=True)
    if force and os.path.exists(CACHED_PATH):
        try:
            os.remove(CACHED_PATH)
        except Exception:
            pass

    if not os.path.exists(CACHED_PATH):
        client = _make_storage_client()
        client.bucket(GCS_BUCKET).blob(GCS_BLOB).download_to_filename(CACHED_PATH)

    return CACHED_PATH

def _ensure_model_path(force: bool = False) -> str:
    if USE_GCS:
        try:
            return _download_model(force=force)
        except Exception as e:
            st.error(f"Failed to download model from gs://{GCS_BUCKET}/{GCS_BLOB}\n\n{e}")
            st.stop()
    else:
        if not os.path.exists(LOCAL_MODEL):
            st.error(
                f"Model file '{LOCAL_MODEL}' not found.\n"
                "• Local run: put best.pt next to this file or set LOCAL_MODEL\n"
                "• Cloud: set USE_GCS='1' and add GCS settings + service account in Secrets"
            )
            st.stop()
        return LOCAL_MODEL

@st.cache_resource(show_spinner=True)
def load_model(model_sig: str):
    """Cached by signature so changing GCS_BLOB forces a new YOLO instance."""
    path = _ensure_model_path(force=False)
    return YOLO(path)

def get_class_names(model: YOLO):
    names = getattr(getattr(model, "model", model), "names", None)
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    if isinstance(names, list):
        return names
    # fallback (TACO-20)
    return [
        "Cigarette","Plastic film","Clear plastic bottle","Other plastic",
        "Other plastic wrapper","Drink can","Plastic bottle cap","Plastic straw",
        "Broken glass","Styrofoam piece","Glass bottle","Disposable plastic cup",
        "Pop tab","Other carton","Paper","Food wrapper","Metal bottle cap",
        "Cardboard","Paper cup","Plastic utensils"
    ]

# =========================================================
# 20-class subset (UI filter only; does not retrain)
# =========================================================
SUBSET_20 = {
    "Cigarette","Plastic film","Clear plastic bottle","Other plastic",
    "Other plastic wrapper","Drink can","Plastic bottle cap","Plastic straw",
    "Broken glass","Styrofoam piece","Glass bottle","Disposable plastic cup",
    "Pop tab","Other carton","Normal paper","Metal bottle cap",
    "Plastic lid","Paper cup","Corrugated carton","Aluminium foil"
}

# =========================================================
# Shibuya mapping (extended for subset names)
# =========================================================
def shibuya_category_and_note(name: str):
    cat, note = "Burnable (default) or check ward", "If plastic is clean → recyclable; if not washable → burnable."
    paper_like = {"Paper", "Normal paper", "Cardboard", "Corrugated carton", "Other carton"}
    plastics = {
        "Plastic film","Other plastic","Other plastic wrapper","Plastic bottle cap",
        "Plastic lid","Plastic straw","Disposable plastic cup","Plastic utensils","Food wrapper","Styrofoam piece"
    }
    if name in {"Cardboard","Corrugated carton"}:
        cat, note = "Recyclable — Cardboard", "Flatten; bundle; no bags."
    elif name in {"Paper","Normal paper","Other carton"}:
        cat, note = "Recyclable — Paper", "Bundle by type; if soiled → burnable."
    elif name == "Clear plastic bottle":
        cat, note = "Recyclable — PET bottle", "Remove cap+label, rinse; crush."
    elif name == "Drink can":
        cat, note = "Recyclable — Can", "Rinse; place in clear bag."
    elif name == "Glass bottle":
        cat, note = "Recyclable — Bottle", "Remove cap; rinse. Heavily soiled/damaged → non-burnable."
    elif name in plastics:
        cat, note = "Recyclable — Plastics (if clean)", "Lightly rinse; non-washable → burnable."
    elif name in {"Metal bottle cap","Pop tab","Aluminium foil"}:
        cat, note = "Non-burnable (small metal)", "Clear/semiclear bag."
    elif name == "Broken glass":
        cat, note = "Non-burnable (glass/ceramic)", "Wrap; mark 'キケン'."
    elif name == "Cigarette":
        cat, note = "Burnable", "Fully extinguish before disposal."
    return cat, note

# =========================================================
# Inference helpers
# =========================================================
def pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))

def run_inference_rgb(rgb: np.ndarray, conf: float, iou: float, imgsz: int, show_only_subset: bool):
    model = load_model(MODEL_SIG)
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
            if show_only_subset and (name not in SUBSET_20):
                continue
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

def process_video(file_bytes: bytes, conf: float, iou: float, imgsz: int,
                  frame_step: int, max_frames: int, show_only_subset: bool):
    import cv2
    in_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    in_file.write(file_bytes); in_file.flush(); in_file.close()

    cap = cv2.VideoCapture(in_file.name)
    if not cap.isOpened():
        st.error("Could not read the uploaded video.")
        return None, {}, {}, 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps   = max(1.0, cap.get(cv2.CAP_PROP_FPS) or 24.0)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_out = max(1.0, fps / max(1, frame_step))
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = cv2.VideoWriter(out_file.name, fourcc, fps_out, (w, h))

    model = load_model(MODEL_SIG)
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
                if show_only_subset and (name not in SUBSET_20):
                    continue
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
st.caption("Loads your YOLO .pt from GCS (or local). Changing GCS_BLOB auto-loads the new model.")

# Debug sidebar
with st.sidebar:
    sa = _read_sa_from_secrets()
    st.header("Debug")
    st.write({
        "USE_GCS": USE_GCS,
        "GCS_BUCKET": GCS_BUCKET,
        "GCS_BLOB": GCS_BLOB,
        "has_service_account": bool(sa),
        "project_from_sa": sa.get("project_id") if sa else None,
        "cached_model_exists": os.path.exists(CACHED_PATH) if USE_GCS else os.path.exists(LOCAL_MODEL),
        "cache_file": CACHED_PATH if USE_GCS else LOCAL_MODEL,
        "model_sig": MODEL_SIG,
    })

    colA, colB = st.columns(2)
    if colA.button("Test model path"):
        p = _ensure_model_path(force=False)
        st.success(f"Model ready at {p} ({os.path.getsize(p)/1e6:.2f} MB)")
    if colB.button("Force re-download"):
        _ensure_model_path(force=True)
        st.success("Re-downloaded model from GCS.")

    colC, colD = st.columns(2)
    if colC.button("Clear model cache"):
        st.cache_resource.clear()
        st.success("Cleared model cache; will reload on next inference.")
    if colD.button("Show model info"):
        try:
            m = load_model(MODEL_SIG)
            names = get_class_names(m)
            task = getattr(getattr(m, "model", m), "task", "unknown")
            st.write({"task": task, "num_classes": len(names), "classes_preview": names[:10]})
        except Exception as e:
            st.error(str(e))

    # GCS browser to find the correct .pt
    st.write("---")
    st.subheader("GCS Browser")
    default_prefix = "taco_env/TACO/derived/_artifacts/weights/"
    pref = st.text_input("Prefix", value=default_prefix)
    if st.button("List .pt under prefix"):
        try:
            names = [n for n in list_gcs(pref) if n.endswith(".pt")]
            subset20 = [n for n in names if "subset20" in n.lower()]
            st.write("All .pt:", names if names else "(none)")
            if subset20:
                st.success("Likely 20-class models:")
                st.write(subset20)
            else:
                st.info("No filenames with 'subset20' found; pick the correct run manually.")
        except Exception as e:
            st.error(str(e))

# Controls
top1, top2, top3, top4 = st.columns(4)
conf  = top1.slider("Confidence", 0.01, 0.90, 0.15, 0.01)
iou   = top2.slider("IoU",        0.10, 0.90, 0.45, 0.01)
imgsz = top3.select_slider("Image size", options=[320, 480, 640, 800, 960, 1024], value=960)
show_only_subset = top4.toggle("Show only 20-class subset", value=True)

mode = st.selectbox("Input type", ("Capture Photo (Webcam)", "Upload Image", "Upload Video"))

if st.button("🔁 Load model now"):
    _ = load_model(MODEL_SIG)
    st.success("Model loaded.")

# Webcam
if mode == "Capture Photo (Webcam)":
    shot = st.camera_input("Take a picture")
    if shot is not None:
        img = Image.open(shot).convert("RGB")
        st.image(img, caption="Captured photo", use_container_width=True)
        if st.button("Run detection on photo"):
            rgb = pil_to_rgb(img)
            dets, counts, vis, names, task = run_inference_rgb(rgb, conf, iou, imgsz, show_only_subset)
            st.caption(f"Model task: {task} • Classes reported by weights: {len(names)}")
            st.subheader("Detections")
            st.image(vis, use_container_width=True)
            if dets:
                st.dataframe(pd.DataFrame(dets))
                by_sh = pd.Series([d["shibuya_type"] for d in dets]).value_counts()
                st.subheader("Counts by Shibuya type")
                st.bar_chart(by_sh)
            else:
                st.info("No detections. Lower confidence / increase image size / try a clearer object.")

# Image
elif mode == "Upload Image":
    up = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if up is not None:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)
        if st.button("Run detection on image"):
            rgb = pil_to_rgb(img)
            dets, counts, vis, names, task = run_inference_rgb(rgb, conf, iou, imgsz, show_only_subset)
            st.caption(f"Model task: {task} • Classes reported by weights: {len(names)}")
            st.subheader("Detections")
            st.image(vis, use_container_width=True)
            if dets:
                st.dataframe(pd.DataFrame(dets))
                by_sh = pd.Series([d["shibuya_type"] for d in dets]).value_counts()
                st.subheader("Counts by Shibuya type")
                st.bar_chart(by_sh)
            else:
                st.info("No detections. Lower confidence / increase image size / try a clearer object.")

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
                    vid.read(), conf=conf, iou=iou, imgsz=imgsz,
                    frame_step=step, max_frames=cap_limit, show_only_subset=show_only_subset
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

