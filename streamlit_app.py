# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import os
from typing import Any, List

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

torch.classes.__path__ = []  # Torch module __path__._path issue: https://github.com/datalab-to/marker/issues/442


class Inference:
    """
    A class to perform object detection, image classification, image segmentation and pose estimation inference.

    This class provides functionalities for loading models, configuring settings, uploading video/image files,
    and capturing camera snapshots in the browser using Streamlit's st.camera_input.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Inference class, checking Streamlit requirements and setting up the model path."""
        check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
        import streamlit as st

        self.st = st
        self.source = None  # "camera", "video", "image"
        self.img_file_names: List[dict] = []  # [{path, name}, ...]
        self.enable_trk = False  # (video only)
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind: List[int] = []
        self.model: YOLO | None = None

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = self.temp_dict.get("model")

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self) -> None:
        """Set up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""
        main_title_cfg = """<div><h1 style="color:#111F68; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""
        sub_title_cfg = """<div><h5 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience object detection on your camera snapshots, videos, and images 
        with the power of Ultralytics YOLO! 🚀</h5></div>"""

        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self) -> None:
        """Configure the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")
        # Replace "webcam" with "camera (snapshot)" that uses st.camera_input
        self.source = self.st.sidebar.selectbox(
            "Source",
            ("camera (snapshot)", "video", "image"),
        )

        # Tracking only makes sense for continuous video
        if self.source == "video":
            self.enable_trk = self.st.sidebar.radio("Enable Tracking (video only)", ("Yes", "No")) == "Yes"

        self.conf = float(self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01))
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))

        # For video preview panes
        if self.source == "video":
            col1, col2 = self.st.columns(2)
            self.org_frame = col1.empty()
            self.ann_frame = col2.empty()

    def source_upload(self) -> None:
        """Handle uploads (video & images). 'camera' uses st.camera_input in its own method."""
        from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS  # scope import

        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=VID_FORMATS)
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())
                with open("ultralytics.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "image":
            import tempfile  # scope import

            imgfiles = self.st.sidebar.file_uploader("Upload Image Files", type=IMG_FORMATS, accept_multiple_files=True)
            if imgfiles:
                for imgfile in imgfiles:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{imgfile.name.split('.')[-1]}") as tf:
                        tf.write(imgfile.read())
                        self.img_file_names.append({"path": tf.name, "name": imgfile.name})

    def configure(self) -> None:
        """Configure the model and load selected classes for inference."""
        M_ORD, T_ORD = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"], ["", "-seg", "-pose", "-obb", "-cls"]
        available_models = sorted(
            [
                x.replace("yolo", "YOLO")
                for x in GITHUB_ASSETS_STEMS
                if any(x.startswith(b) for b in M_ORD) and "grayscale" not in x
            ],
            key=lambda x: (M_ORD.index(x[:7].lower()), T_ORD.index(x[7:].lower() or "")),
        )
        if self.model_path:
            available_models.insert(0, self.model_path)
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            if (
                selected_model.endswith((".pt", ".onnx", ".torchscript", ".mlpackage", ".engine"))
                or "openvino_model" in selected_model
            ):
                model_path = selected_model
            else:
                model_path = f"{selected_model.lower()}.pt"
            self.model = YOLO(model_path)
            class_names = list(self.model.names.values())
        self.st.success("Model loaded successfully!")

        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]
        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)

    def image_inference(self) -> None:
        """Perform inference on uploaded images."""
        for img_info in self.img_file_names:
            img_path = img_info["path"]
            image = cv2.imread(img_path)
            if image is not None:
                self.st.markdown(f"#### Processed: {img_info['name']}")
                col1, col2 = self.st.columns(2)
                with col1:
                    self.st.image(image, channels="BGR", caption="Original Image")
                results = self.model(image, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_image = results[0].plot()
                with col2:
                    self.st.image(annotated_image, channels="BGR", caption="Predicted Image")
                try:
                    os.unlink(img_path)
                except FileNotFoundError:
                    pass
            else:
                self.st.error("Could not load the uploaded image.")

    def camera_inference(self) -> None:
        """Capture a snapshot from the user's browser camera and run inference."""
        self.st.markdown("### Camera")
        # The widget returns an UploadedFile after the user clicks "Take Photo"
        cam_file = self.st.camera_input("Take a photo")
        if cam_file is None:
            self.st.info("Click **Take Photo** to capture an image from your camera.")
            return

        # Convert to OpenCV BGR
        pil_img = Image.open(cam_file)
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Inference
        results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
        annotated = results[0].plot()

        # Show side-by-side
        c1, c2 = self.st.columns(2)
        c1.image(pil_img, caption="Original")
        c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Predicted")

    def video_inference(self) -> None:
        """Perform inference on uploaded video file."""
        if not self.vid_file_name:
            self.st.info("Please upload a video file to perform inference.")
            return

        cap = cv2.VideoCapture(self.vid_file_name)
        if not cap.isOpened():
            self.st.error("Could not open video source.")
            return

        stop_button = self.st.sidebar.button("Stop")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                self.st.warning("End of video or failed to read frame.")
                break

            if self.enable_trk:
                results = self.model.track(
                    frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                )
            else:
                results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)

            annotated_frame = results[0].plot()

            if stop_button:
                cap.release()
                self.st.stop()

            self.org_frame.image(frame, channels="BGR", caption="Original Frame")
            self.ann_frame.image(annotated_frame, channels="BGR", caption="Predicted Frame")

        cap.release()

    def inference(self) -> None:
        """Main entry: set up UI, load model, and route to selected source."""
        self.web_ui()
        self.sidebar()
        self.source_upload()
        self.configure()

        # Start button gates the heavy work for video/images; camera widget can live inside too
        if self.st.sidebar.button("Start"):
            if self.source == "image":
                if self.img_file_names:
                    self.image_inference()
                else:
                    self.st.info("Please upload one or more images.")
            elif self.source == "video":
                self.video_inference()
            else:  # "camera (snapshot)"
                self.camera_inference()

        # For better UX you can also show the camera widget even before Start:
        if self.source == "camera (snapshot)":
            self.camera_inference()


if __name__ == "__main__":
    import sys
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None
    Inference(model=model).inference()