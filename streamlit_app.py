import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="When AI Sees Litter — Shibuya", page_icon="♻️", layout="wide")

# ======================= THEME (Light agriculture vibe) =======================
def apply_agri_theme():
    st.markdown("""
    <style>
      :root{
        --agri-primary:#79C16D;       /* fresh leaf green */
        --agri-primary-dark:#4FA25A;  /* deeper leaf green */
        --agri-accent:#CFEAC0;        /* soft lime highlight */
        --agri-bg:#FAFEF6;            /* warm off-white/green */
        --agri-card:#FFFFFF;          /* pure white cards */
        --agri-text:#0F2A1C;          /* deep green text */
        --agri-muted:#6F8B7A;         /* muted copy */
        --agri-pill:#EEF7E9;          /* pill bg */
        --agri-border:#E5EFE3;        /* subtle borders */
      }
      html, body, [data-testid="stAppViewContainer"]{
        background: var(--agri-bg);
        color: var(--agri-text);
      }
      .main .block-container{ padding-top: 1rem !important; }

      .hero{
        background:
          radial-gradient(700px 280px at 10% -20%, rgba(121,193,109,.18), transparent),
          linear-gradient(135deg, #FFFFFF 0%, #F7FBF2 100%);
        border:1px solid var(--agri-border);
        border-radius:24px; padding:22px 20px; margin: 6px 0 12px 0;
      }
      .hero h1{ margin:0 0 6px 0; font-weight:900; letter-spacing:.2px; font-size:1.8rem; }
      .hero p{ margin:0 0 10px 0; color: var(--agri-muted); }
      .pill{ display:inline-block; background: var(--agri-pill); padding: 2px 10px 4px 10px;
             border-radius: 999px; color: var(--agri-primary-dark); border:1px solid var(--agri-border); }

      .section{ margin: 10px 0 18px 0; padding:16px; background: var(--agri-card);
                border:1px solid var(--agri-border); border-radius: 18px; }

      .eco-card{ background:#FFFFFF; border:1px solid var(--agri-border); border-radius:22px;
                 padding:18px 16px; margin:10px 0 18px 0; box-shadow: 0 3px 16px rgba(0,0,0,.04); }
      .eco-head{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }
      .eco-emoji{ font-size:1.5rem; }
      .eco-title{ font-weight:900; font-size:1.28rem; }
      .eco-badge{ margin-left:auto; background: var(--agri-pill); color: var(--agri-primary-dark);
                  border:1px solid var(--agri-border); border-radius:999px; padding:4px 10px; font-size:.85rem; }
      .eco-meta{ margin: 6px 0 8px 0; color: var(--agri-muted); font-size:.95rem; }
      .eco-section-title{ font-weight:800; margin-top:8px; margin-bottom:4px; }
      .eco-list{ margin:0 0 4px 0; padding-left:18px;}
      .eco-list li{ margin: 2px 0; }
      .chip-row{ display:flex; flex-wrap:wrap; gap:8px; margin: 6px 0 2px 0; }
      .chip{ background: var(--agri-pill); color: var(--agri-primary-dark); border:1px solid var(--agri-border);
             border-radius:999px; padding:4px 10px; font-size:.88rem; }
      .eco-links{ display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }
      .eco-link{ border-radius:999px; padding:8px 12px; border:1px solid var(--agri-border);
                 background:#fff; text-decoration:none !important; color: var(--agri-primary-dark) !important; font-weight:700; }
      .eco-link:hover{ background: var(--agri-pill); }

      .howto li{ margin:2px 0; }
      .sdg-row{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
      .sdg-card{ display:flex; gap:10px; align-items:center; border:1px solid var(--agri-border);
                 background:#fff; padding:10px 12px; border-radius:14px; }
      .sdg-card img{ width:54px; height:auto; }
      .sdg-card .txt{ font-weight:700; }
      .link-chips{ display:flex; flex-wrap:wrap; gap:10px; margin-top:8px; }
      .link-chip{ border:1px solid var(--agri-border); padding:8px 12px; border-radius:999px;
                  text-decoration:none; color: var(--agri-primary-dark); background:#fff; font-weight:700; }
      .link-chip:hover{ background: var(--agri-pill); }
    </style>
    """, unsafe_allow_html=True)

apply_agri_theme()

# ======================= Config & Model =======================
MODEL_URL     = os.getenv("MODEL_URL", "https://raw.githubusercontent.com/Bellzum/streamlit-main/main/yolo_tue_3classes.pt")
LOCAL_MODEL   = os.getenv("LOCAL_MODEL", "best.pt")
CACHED_PATH   = "/tmp/models/best.pt"
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))

CLASS_NAMES   = ["Clear plastic bottle", "Drink can", "Plastic bottle cap"]
IMGSZ_OPTIONS = [320, 416, 512, 640, 800, 960, 1280]

# ======================= Links + feature images (optional) =======================
FEATURE_LINKS = {
    "cleaner": os.getenv("LINK_CLEANER", "https://www.city.shibuya.tokyo.jp/contents/living-in-shibuya/en/daily/garbage.html"),
    "recycling": os.getenv("LINK_RECYCLING", "https://www.petbottle-rec.gr.jp/english/actual.html"),
    "insights": os.getenv("LINK_INSIGHTS", "#"),
}
FEATURE_IMAGES = {
    "cleaner": "assets/cleaner.jpg",
    "recycling": "assets/recycling.jpg",
    "insights": "assets/insights.jpg",
}
def resolve_asset(path: str):
    if not path: return None
    if os.path.exists(path): return path
    base = os.path.basename(path)
    alt = os.path.join("assets", base)
    return alt if os.path.exists(alt) else None

# ======================= Official Shibuya references =======================
SHIBUYA_GUIDE_URL = "https://www.city.shibuya.tokyo.jp/contents/living-in-shibuya/en/daily/garbage.html"
SHIBUYA_POSTER_EN = "https://files.city.shibuya.tokyo.jp/assets/12995aba8b194961be709ba879857f70/bfda2f5d763343b5a0b454087299d57f/2024wakedashiEnglish.pdf#page=2"
SHIBUYA_PLASTICS_NOTICE = "https://files.city.shibuya.tokyo.jp/assets/12995aba8b194961be709ba879857f70/0cdf099fdfe8456fbac12bb5ad7927e4/assets_kusei_ShibuyaCityNews2206_e.pdf#page=1"

# PET step photos (Fukuoka City — illustrates same steps)
FUKUOKA_PET_STEPS = [
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph04.png",  # remove cap
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph05.png",  # remove label
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph06.png",  # rinse
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph07.png",  # crush
]

# Recycling marks (Wikipedia Commons PNG thumbnails)
ICON_PET   = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Recycling_pet.svg/120px-Recycling_pet.svg.png"
ICON_AL    = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Recycling_alumi.svg/120px-Recycling_alumi.svg.png"
ICON_STEEL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Recycling_steel.svg/120px-Recycling_steel.svg.png"
ICON_PLA   = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Recycling_pla.svg/120px-Recycling_pla.svg.png"

# SDG icon images (official UN files)
SDG_11 = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Sustainable_Development_Goal_11SustainableCities.svg/500px-Sustainable_Development_Goal_11SustainableCities.svg.png"
SDG_12 = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/Sustainable_Development_Goal_12ResponsibleConsumption.svg/500px-Sustainable_Development_Goal_12ResponsibleConsumption.svg.png"
SDG_13 = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Sustainable_Development_Goal_13Climate.svg/600px-Sustainable_Development_Goal_13Climate.svg.png"

# Carbon-credit helpful links
LINK_UN_CNP  = "https://unfccc.int/climate-action/united-nations-carbon-offset-platform"
LINK_UN_CNP2 = "https://offset.climateneutralnow.org/"
LINK_WB_MRV  = "https://www.worldbank.org/en/news/feature/2022/07/27/what-you-need-to-know-about-the-measurement-reporting-and-verification-mrv-of-carbon-credits"
LINK_GS      = "https://www.goldstandard.org/"
LINK_VERRA   = "https://verra.org/programs/verified-carbon-standard/"
LINK_JCREDIT = "https://japancredit.go.jp/english/"

# Hanwa & CCBJI can-to-can flow images
HANWA_CAN2CAN = "https://www.hanwa.co.jp/images/csr/business/img_5_01.png"
CCBJI_CAN2CAN = "https://en.ccbji.co.jp/upload/images/20221222-1-1(5).jpg"

# ======================= Guidance content =======================
GUIDE = {
    "Clear plastic bottle": {
        "title": "Shibuya disposal: PET bottle (resource)",
        "emoji": "🧴",
        "materials": "Bottle body is PET (polyethylene terephthalate). Caps and labels are PP/PE.",
        "why_separate": [
            "Caps and labels (PP/PE) contaminate the PET stream if left on.",
            "Shibuya asks you to remove caps and labels and sort them with Plastics."
        ],
        "steps": [
            "Remove the cap and label.",
            "Rinse the bottle.",
            "Crush it flat.",
            "Put PET bottles in a transparent bag for PET.",
            "Put caps and labels with Plastics."
        ],
        "recycles_to": ["New PET bottles", "Fibers for clothing and bags", "Sheets/films"],
        "facts": [
            {
                "text": "Japan’s reported plastic 'recycling' rate includes thermal recovery; clean PET enables high-value bottle-to-bottle.",
                "url": "https://japan-forward.com/japans-plastic-recycling-the-unseen-reality/"
            },
            {
                "text": "Recycled PET in Japan becomes new bottles, sheets and fibers for clothing/bags.",
                "url": "https://www.petbottle-rec.gr.jp/english/actual.html"
            }
        ],
        "images": FUKUOKA_PET_STEPS,
        "icons": [ICON_PET],
        "link": SHIBUYA_GUIDE_URL,
        "poster": SHIBUYA_POSTER_EN,
    },
    "Drink can": {
        "title": "Shibuya disposal: Aluminum or steel can (resource)",
        "emoji": "🥫",
        "materials": "Mostly aluminum; some cans are steel.",
        "why_separate": [
            "Clean metal cans keep a high-value recycling stream.",
            "Aluminum recycling saves major energy vs producing new metal."
        ],
        "steps": [
            "Rinse the can.",
            "Optional: Lightly crush/squeeze to save space (only if your building/bin instructions allow).",
            "Put cans in a transparent bag for cans."
        ],
        "recycles_to": [
            "New beverage cans (can-to-can)",
            "Automotive & construction parts (aluminum)",
            "Remelt scrap ingots"
        ],
        "facts": [
            {
                "text": "Coca-Cola Bottlers Japan promotes CAN-to-CAN, including products using recycled aluminum bodies.",
                "url": "https://en.ccbji.co.jp/news/detail.php?id=1347"
            },
            {
                "text": "Hanwa: used aluminum cans are cleaned, melted and supplied as remelt scrap ingots to aluminum mills — then used again as cans.",
                "url": HANWA_CAN2CAN
            }
        ],
        # Show both Hanwa and CCBJI flow images on the card
        "images": [HANWA_CAN2CAN, CCBJI_CAN2CAN],
        "icons": [ICON_AL, ICON_STEEL],
        "link": SHIBUYA_GUIDE_URL,
        "poster": SHIBUYA_POSTER_EN,
    },
    "Plastic bottle cap": {
        "title": "Shibuya disposal: Plastic bottle cap (plastic item)",
        "emoji": "🔘",
        "materials": "PP or PE (polypropylene or polyethylene) closures.",
        "why_separate": [
            "Caps are not PET. Separating avoids contaminating bottle-to-bottle recycling.",
            "In Shibuya, caps & labels go with Plastic items (プラ), not with PET bottles."
        ],
        "steps": ["Remove from the bottle.", "Rinse if sticky.", "Put caps with Plastic items in a clear/semi-clear bag."],
        "recycles_to": ["New caps (pilots)", "Plastic containers/packaging", "Pallets & molded goods"],
        "facts": [
            {
                "text": "Separating PP/PE caps and labels keeps the PET stream clean for high-value recycling.",
                "url": "https://japan-forward.com/japans-plastic-recycling-the-unseen-reality/"
            }
        ],
        "images": [],
        "icons": [ICON_PLA],
        "link": SHIBUYA_GUIDE_URL,
        "poster": SHIBUYA_PLASTICS_NOTICE,
    },
}

def _guide_link(url: str, label: str):
    st.markdown(f'<a class="eco-link" href="{url}" target="_blank" rel="noopener">{label}</a>', unsafe_allow_html=True)

def _guidance_text(info: dict):
    if info.get("materials"):
        st.markdown(f'<div class="eco-meta"><strong>Material:</strong> {info["materials"]}</div>', unsafe_allow_html=True)
    if info.get("why_separate"):
        st.markdown('<div class="eco-section-title">Why separate?</div>', unsafe_allow_html=True)
        st.markdown('<ul class="eco-list">', unsafe_allow_html=True)
        for reason in info["why_separate"]:
            st.markdown(f'<li>{reason}</li>', unsafe_allow_html=True)
        st.markdown('</ul>', unsafe_allow_html=True)

    st.markdown('<div class="eco-section-title">How to put out</div>', unsafe_allow_html=True)
    st.markdown('<ul class="eco-list">', unsafe_allow_html=True)
    for step in info["steps"]:
        st.markdown(f'<li>{step}</li>', unsafe_allow_html=True)
    st.markdown('</ul>', unsafe_allow_html=True)

    if info.get("recycles_to"):
        st.markdown('<div class="eco-section-title">Commonly recycled into</div>', unsafe_allow_html=True)
        st.markdown('<div class="chip-row">', unsafe_allow_html=True)
        for item in info["recycles_to"]:
            st.markdown(f'<div class="chip">{item}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    facts = info.get("facts", [])
    if facts:
        st.markdown('<div class="eco-section-title">Did you know?</div>', unsafe_allow_html=True)
        st.markdown('<ul class="eco-list">', unsafe_allow_html=True)
        for fact in facts:
            st.markdown(f'<li>{fact["text"]}</li>', unsafe_allow_html=True)
        st.markdown('</ul>', unsafe_allow_html=True)
        st.markdown('<div class="eco-links">', unsafe_allow_html=True)
        for fact in facts:
            _guide_link(fact["url"], "Learn more")
        st.markdown('</div>', unsafe_allow_html=True)

def show_shibuya_guidance(label: str, count: int = 0):
    info = GUIDE.get(label)
    if not info:
        return
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)
    st.markdown(f"""
      <div class="eco-head">
        <div class="eco-emoji">{info['emoji']}</div>
        <div class="eco-title">{info['title']}</div>
        <div class="eco-badge">Detected: {count}</div>
      </div>
    """, unsafe_allow_html=True)

    # Icons (recycling marks)
    if info.get("icons"):
        st.image(info["icons"], width=48, caption=[""]*len(info["icons"]))

    # Images (PET steps or flow diagrams)
    imgs = info.get("images") or []
    if imgs:
        left, right = st.columns([1, 2], vertical_alignment="center")
        with left:
            if len(imgs) == 1:
                st.image(imgs[0], use_container_width=True)   # large single
            elif len(imgs) <= 3:
                for im in imgs:
                    st.image(im, use_container_width=True)   # stack a few, readable
            else:
                st.image(imgs, width=160, caption=[""]*len(imgs))  # many (thumbnail)
        with right:
            _guidance_text(info)
    else:
        _guidance_text(info)

    # Links
    st.markdown('<div class="eco-links">', unsafe_allow_html=True)
    if info.get("poster"): _guide_link(info["poster"], "Open Shibuya poster")
    _guide_link(info["link"], "Official Shibuya guidance (site)")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================= Helpers =======================
def _download_file(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
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
        st.error("Model file not found. Provide MODEL_URL or place best.pt next to this file.")
        st.stop()
    return LOCAL_MODEL

def _cache_key_for(path: str) -> str:
    try: return f"{path}:{os.path.getmtime(path)}:{os.path.getsize(path)}"
    except Exception: return path

@st.cache_resource(show_spinner=True)
def _load_model_cached(path: str, key: str):
    return YOLO(path)

def load_model():
    path = _ensure_model_path()
    return _load_model_cached(path, _cache_key_for(path))

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return arr[:, :, ::-1]

# Robust label drawing that avoids clipping
def draw_boxes(bgr, dets):
    import cv2
    out = bgr.copy()
    H, W = out.shape[:2]
    color = (28,160,78)  # theme green (BGR)

    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f'{d["class_name"]} {d["score"]:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, thick = 0.5, 1
        (tw, th), _ = cv2.getTextSize(label, font, fs, thick)

        y_text = y1 - 4
        if y_text - th - 4 < 0:
            y_text = min(y1 + th + 6, H - 2)

        x_text = max(0, min(x1, W - tw - 6))

        x_bg1, y_bg1 = x_text, max(0, y_text - th - 4)
        x_bg2, y_bg2 = min(x_text + tw + 6, W - 1), min(y_text + 2, H - 1)
        cv2.rectangle(out, (x_bg1, y_bg1), (x_bg2, y_bg2), color, -1)
        cv2.putText(out, label, (x_text + 3, y_text - 2), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

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

# ======================= HEADER =======================
logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
with logo_col:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
with title_col:
    st.markdown("<div style='font-weight:800; font-size:1.1rem;'>When AI Sees Litter — Shibuya</div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Scan litter. Get Shibuya sorting guidance.</h1>
  <p><span class="pill">Quick Detect</span> works on PET bottles, drink cans, and plastic bottle caps.</p>
</div>
""", unsafe_allow_html=True)

# ======================= QUICK DETECT (TOP) =======================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("#### How to use")
st.markdown("""
<ol class="howto">
  <li>Select <b>Upload image</b> (or open your <b>Camera</b>).</li>
  <li>Tap <b>Run detection</b>.</li>
  <li>Follow the card(s) below for Shibuya disposal steps.</li>
</ol>
""", unsafe_allow_html=True)

# Defaults (minimum filters)
_MIN_CONF = 0.05; _MIN_IOU = 0.10; _MIN_IMGSZ = _closest_size(DEFAULT_IMGSZ, IMGSZ_OPTIONS)
_MIN_BOTTLE = 0.00; _MIN_CAN = 0.00; _MIN_CAP = 0.00; _MIN_AREA_PCT = 0.0; _MIN_TTA = False

with st.expander("Advanced settings (optional)"):
    preset = st.radio("Preset", ["Minimum filters", "Recommended", "Strict"], index=0, horizontal=True)
    conf = _MIN_CONF; iou = _MIN_IOU; imgsz = _MIN_IMGSZ
    bottle_min = _MIN_BOTTLE; can_min = _MIN_CAN; cap_min = _MIN_CAP
    min_area_pct = _MIN_AREA_PCT; tta = _MIN_TTA
    if preset == "Recommended":
        conf = 0.25; iou = 0.45; bottle_min = 0.60; can_min = 0.55; cap_min = 0.65; min_area_pct = 0.3; tta = False
    elif preset == "Strict":
        conf = 0.35; iou = 0.50; bottle_min = 0.70; can_min = 0.70; cap_min = 0.75; min_area_pct = 0.5; tta = False
    conf = st.slider("Base confidence", 0.05, 0.95, conf, 0.01, help="Model confidence threshold.")
    iou  = st.slider("IoU", 0.10, 0.90, iou, 0.01)
    imgsz = st.select_slider("Inference image size", options=IMGSZ_OPTIONS, value=_closest_size(int(imgsz), IMGSZ_OPTIONS))
    c1, c2, c3, c4 = st.columns(4)
    bottle_min = c1.slider("Min conf: Bottle", 0.0, 1.0, bottle_min, 0.01)
    can_min    = c2.slider("Min conf: Can",    0.0, 1.0, can_min, 0.01)
    cap_min    = c3.slider("Min conf: Cap",    0.0, 1.0, cap_min, 0.01)
    min_area_pct = c4.slider("Min box area (%)", 0.0, 5.0, min_area_pct, 0.1, help="Ignore tiny boxes by percent of image area.")
    tta = st.toggle("Test time augmentation", value=tta, help="Slower. Sometimes reduces false positives.")

# If Advanced is closed, still use minimums
if "conf" not in locals():
    conf = _MIN_CONF; iou = _MIN_IOU; imgsz = _MIN_IMGSZ
    bottle_min = _MIN_BOTTLE; can_min = _MIN_CAN; cap_min = _MIN_CAP
    min_area_pct = _MIN_AREA_PCT; tta = _MIN_TTA

# Input controls (default = Upload image)
src = st.radio("Input source", ["Upload image", "Camera"], index=0, horizontal=True)
image = None
if src == "Upload image":
    up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if up: image = Image.open(up).convert("RGB")
else:
    shot = st.camera_input("Open your camera", key="cam1")
    if shot: image = Image.open(shot).convert("RGB")

# Model loader (optional)
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

# Show chosen image and run detection
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

            per_class_min = {"Clear plastic bottle": bottle_min, "Drink can": can_min, "Plastic bottle cap": cap_min}
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

                if s < per_class_min.get(name, conf):   continue
                if area < min_area:                      continue

                dets.append({"xyxy": [x1, y1, x2, y2], "class_id": c, "class_name": name, "score": s})
                counts[name] = counts.get(name, 0) + 1

            if not dets:
                st.info("All detections were filtered by thresholds. Try lowering per-class thresholds or min box area.")
            else:
                vis_img = draw_boxes(bgr, dets)
                st.subheader("Detections")
                st.image(vis_img, use_container_width=True)

                # Debug (collapsed)
                with st.expander("Raw detections (debug)", expanded=False):
                    st.dataframe(pd.DataFrame(dets))
                if counts:
                    with st.expander("Counts (debug)", expanded=False):
                        st.bar_chart(pd.Series(counts).sort_values(ascending=False))

                # Guidance cards
                detected_labels = sorted({d["class_name"] for d in dets})
                guide_labels = [lbl for lbl in detected_labels if lbl in GUIDE]
                if guide_labels:
                    st.subheader("Disposal instructions for Shibuya")
                    for lbl in guide_labels:
                        show_shibuya_guidance(lbl, counts.get(lbl, 0))
                else:
                    st.caption("No Shibuya guidance to show for these detections.")
st.markdown('</div>', unsafe_allow_html=True)  # end section

# ======================= Impact & SDGs =======================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("#### Impact & SDGs")

st.markdown("""
- **Carbon credits (what they are):** A carbon credit represents **1 tonne of CO₂-equivalent** reduced or removed. Credits exist only when a **registered project** follows an **approved methodology** and passes **MRV**; they are then **issued on a registry** (e.g., Gold Standard, Verra, or Japan’s J-Credit).  
- **This app does not issue credits.** It helps people sort properly. You may show **educational CO₂e-avoided estimates**, but that is **not** the same as credits.
""")

st.markdown(
    f"""
<div class="link-chips">
  <a class="link-chip" href="{LINK_UN_CNP}" target="_blank" rel="noopener">UN Carbon Offset Platform</a>
  <a class="link-chip" href="{LINK_UN_CNP2}" target="_blank" rel="noopener">Climate Neutral Now (shop credits)</a>
  <a class="link-chip" href="{LINK_WB_MRV}" target="_blank" rel="noopener">World Bank: MRV & 1 credit = 1 tCO₂e</a>
  <a class="link-chip" href="{LINK_GS}" target="_blank" rel="noopener">Gold Standard (program)</a>
  <a class="link-chip" href="{LINK_VERRA}" target="_blank" rel="noopener">Verra VCS (program)</a>
  <a class="link-chip" href="{LINK_JCREDIT}" target="_blank" rel="noopener">Japan J-Credit (official)</a>
</div>
""",
    unsafe_allow_html=True
)

st.markdown("**Our SDGs focus:**")
sdg_html = f"""
<div class="sdg-row">
  <div class="sdg-card">
    <img src="{SDG_12}" alt="SDG 12 icon">
    <div class="txt">12 Responsible Consumption &amp; Production</div>
  </div>
  <div class="sdg-card">
    <img src="{SDG_11}" alt="SDG 11 icon">
    <div class="txt">11 Sustainable Cities &amp; Communities</div>
  </div>
  <div class="sdg-card">
    <img src="{SDG_13}" alt="SDG 13 icon">
    <div class="txt">13 Climate Action</div>
  </div>
 
</div>
"""
st.markdown(sdg_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
