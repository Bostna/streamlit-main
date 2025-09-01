<<<<<<< HEAD
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
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "yolo_litterv1.pt")
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
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import os
from typing import Any, List

import cv2
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

torch.classes.__path__ = []  # Torch module __path__._path issue: https://github.com/datalab-to/marker/issues/442


class Inference:
    """
    A class to perform object detection, image classification, image segmentation and pose estimation inference.

    This class provides functionalities for loading models, configuring settings, uploading video files, and performing
    real-time inference using Streamlit and Ultralytics YOLO models.

    Attributes:
        st (module): Streamlit module for UI creation.
        temp_dict (dict): Temporary dictionary to store the model path and other configuration.
        model_path (str): Path to the loaded model.
        model (YOLO): The YOLO model instance.
        source (str): Selected video source (webcam or video file).
        enable_trk (bool): Enable tracking option.
        conf (float): Confidence threshold for detection.
        iou (float): IoU threshold for non-maximum suppression.
        org_frame (Any): Container for the original frame to be displayed.
        ann_frame (Any): Container for the annotated frame to be displayed.
        vid_file_name (str | int): Name of the uploaded video file or webcam index.
        selected_ind (List[int]): List of selected class indices for detection.

    Methods:
        web_ui: Set up the Streamlit web interface with custom HTML elements.
        sidebar: Configure the Streamlit sidebar for model and inference settings.
        source_upload: Handle video file uploads through the Streamlit interface.
        configure: Configure the model and load selected classes for inference.
        inference: Perform real-time object detection inference.

    Examples:
        Create an Inference instance with a custom model
        >>> inf = Inference(model="path/to/model.pt")
        >>> inf.inference()

        Create an Inference instance with default settings
        >>> inf = Inference()
        >>> inf.inference()
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Inference class, checking Streamlit requirements and setting up the model path.

        Args:
            **kwargs (Any): Additional keyword arguments for model configuration.
        """
        check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
        import streamlit as st

        self.st = st  # Reference to the Streamlit module
        self.source = None  # Video source selection (webcam or video file)
        self.img_file_names = []  # List of image file names
        self.enable_trk = False  # Flag to toggle object tracking
        self.conf = 0.25  # Confidence threshold for detection
        self.iou = 0.45  # Intersection-over-Union (IoU) threshold for non-maximum suppression
        self.org_frame = None  # Container for the original frame display
        self.ann_frame = None  # Container for the annotated frame display
        self.vid_file_name = None  # Video file name or webcam index
        self.selected_ind: List[int] = []  # List of selected class indices for detection
        self.model = None # YOLO model instance

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = str("new_taco1.pt")  # Model file path
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self) -> None:
        """Set up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style

        # Main title of streamlit application
        main_title_cfg = """<div><h1 style="color:#111F68; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""

        # Subtitle of streamlit application
        sub_title_cfg = """<div><h5 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam, videos, and images 
        with the power of Ultralytics YOLO! 🚀</h5></div>"""

        # Set html page configuration and append custom HTML
        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self) -> None:
        """Configure the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:  # Add Ultralytics LOGO
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")  # Add elements to vertical setting menu
        self.source = self.st.sidebar.selectbox(
            "Source",
            ("webcam", "video", "image"),
        )  # Add source selection dropdown
        if self.source in ["webcam", "video"]:
            self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No")) == "Yes"  # Enable object tracking
        self.conf = float(
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
        )  # Slider for confidence
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))  # Slider for NMS threshold

        if self.source != "image":  # Only create columns for video/webcam
            col1, col2 = self.st.columns(2)  # Create two columns for displaying frames
            self.org_frame = col1.empty()  # Container for original frame
            self.ann_frame = col2.empty()  # Container for annotated frame

    def source_upload(self) -> None:
        """Handle video file uploads through the Streamlit interface."""
        from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS  # scope import

        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=VID_FORMATS)
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())  # BytesIO Object
                with open("ultralytics.mp4", "wb") as out:  # Open temporary file as bytes
                    out.write(g.read())  # Read bytes into file
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0  # Use webcam index 0
        elif self.source == "image":
            import tempfile  # scope import

            imgfiles = self.st.sidebar.file_uploader("Upload Image Files", type=IMG_FORMATS, accept_multiple_files=True)
            if imgfiles:
                for imgfile in imgfiles:  # Save each uploaded image to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{imgfile.name.split('.')[-1]}") as tf:
                        tf.write(imgfile.read())
                        self.img_file_names.append({"path": tf.name, "name": imgfile.name})

    def configure(self) -> None:
        """Configure the model and load selected classes for inference."""

        # Add dropdown menu for model selection
        M_ORD, T_ORD = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"], ["", "-seg", "-pose", "-obb", "-cls"]
        available_models = sorted(
            [
                x.replace("yolo", "YOLO")
                for x in GITHUB_ASSETS_STEMS
                if any(x.startswith(b) for b in M_ORD) and "grayscale" not in x
            ],
            key=lambda x: (M_ORD.index(x[:7].lower()), T_ORD.index(x[7:].lower() or "")),
        )
        if self.model_path:  # Insert user provided custom model in available_models
            available_models.insert(0, self.model_path)
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            if (
                selected_model.endswith((".pt", ".onnx", ".torchscript", ".mlpackage", ".engine"))
                or "openvino_model" in selected_model
            ):
                model_path = selected_model
            else:
                model_path = f"{selected_model.lower()}.pt"  # Default to .pt if no model provided during function call.
            self.model = YOLO(model_path)  # Load the YOLO model
            class_names = list(self.model.names.values())  # Convert dictionary to list of class names
        self.st.success("Model loaded successfully!")

        # Multiselect box with class names and get indices of selected classes
        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):  # Ensure selected_options is a list
            self.selected_ind = list(self.selected_ind)

    def image_inference(self) -> None:
        """Perform inference on uploaded images."""
        for idx, img_info in enumerate(self.img_file_names):
            img_path = img_info["path"]
            image = cv2.imread(img_path)  # Load and display the original image
            if image is not None:
                self.st.markdown(f"#### Processed: {img_info['name']}")
                col1, col2 = self.st.columns(2)
                with col1:
                    self.st.image(image, channels="BGR", caption="Original Image")
                results = self.model(image, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_image = results[0].plot()
                with col2:
                    self.st.image(annotated_image, channels="BGR", caption="Predicted Image")
                try:  # Clean up temporary file
                    os.unlink(img_path)
                except FileNotFoundError:
                    pass  # File doesn't exist, ignore
            else:
                self.st.error("Could not load the uploaded image.")

    def inference(self) -> None:
        """Perform real-time object detection inference on video or webcam feed."""
        self.web_ui()  # Initialize the web interface
        self.sidebar()  # Create the sidebar
        self.source_upload()  # Upload the video source
        self.configure()  # Configure the app

        if self.st.sidebar.button("Start"):
            if self.source == "image":
                if self.img_file_names:
                    self.image_inference()
                else:
                    self.st.info("Please upload an image file to perform inference.")
                return

            stop_button = self.st.sidebar.button("Stop")  # Button to stop the inference
            cap = cv2.VideoCapture(self.vid_file_name)  # Capture the video
            if not cap.isOpened():
                self.st.error("Could not open webcam or video source.")
                return

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    self.st.warning("Failed to read frame from webcam. Please verify the webcam is connected properly.")
                    break

                # Process frame with model
                if self.enable_trk:
                    results = self.model.track(
                        frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                    )
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)

                annotated_frame = results[0].plot()  # Add annotations on frame

                if stop_button:
                    cap.release()  # Release the capture
                    self.st.stop()  # Stop streamlit app

                self.org_frame.image(frame, channels="BGR", caption="Original Frame")  # Display original frame
                self.ann_frame.image(annotated_frame, channels="BGR", caption="Predicted Frame")  # Display processed

            cap.release()  # Release the capture
        cv2.destroyAllWindows()  # Destroy all OpenCV windows


if __name__ == "__main__":
    import sys  # Import the sys module for accessing command-line arguments

    # Check if a model name is provided as a command-line argument
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # Assign first argument as the model name if provided
    # Create an instance of the Inference class and run inference
    Inference(model=model).inference()
>>>>>>> b_note
