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

      /* Top bar note (optional) */
      .topnote{
        background: linear-gradient(180deg, rgba(121,193,109,.12), rgba(121,193,109,0));
        border:1px solid var(--agri-border);
        border-radius: 0 0 16px 16px;
        padding: 8px 14px;
        text-align:center;
        color: var(--agri-muted);
        font-size:.92rem;
        margin-bottom: 8px;
      }

      /* Hero */
      .hero{
        background:
          radial-gradient(700px 280px at 10% -20%, rgba(121,193,109,.18), transparent),
          linear-gradient(135deg, #FFFFFF 0%, #F7FBF2 100%);
        border:1px solid var(--agri-border);
        border-radius:24px;
        padding:26px 22px;
        margin: 6px 0 18px 0;
      }
      .hero h1{
        margin:0 0 6px 0;
        font-weight:900; letter-spacing:.2px;
        font-size:2.0rem;
      }
      .hero p{
        margin:0 0 14px 0; color: var(--agri-muted);
      }
      .pill{
        display:inline-block;
        background: var(--agri-pill);
        padding: 2px 10px 4px 10px;
        border-radius: 999px;
        color: var(--agri-primary-dark);
        border:1px solid var(--agri-border);
      }
      .cta-row{ display:flex; gap:10px; flex-wrap:wrap }
      .cta{
        background: var(--agri-primary);
        color: #fff;
        padding:10px 14px;
        border-radius: 999px;
        font-weight:800;
        text-decoration:none !important;
        border:1px solid color-mix(in srgb, var(--agri-primary) 60%, #fff);
      }
      .cta.secondary{
        background: #fff;
        color: var(--agri-primary-dark);
        border:1px solid var(--agri-border);
      }
      .cta:hover{ filter: brightness(0.97); }

      /* Section shells */
      .section{
        margin: 10px 0 22px 0;
        padding:18px;
        background: var(--agri-card);
        border:1px solid var(--agri-border);
        border-radius: 20px;
      }

      /* KPI cards */
      .kpi{
        text-align:center;
        border:1px solid var(--agri-border);
        padding:16px;
        border-radius:18px;
        background: #FFFFFF;
        box-shadow: 0 3px 16px rgba(0,0,0,.03);
      }
      .kpi .big{ font-size:1.7rem; font-weight:900; line-height:1.2; }
      .kpi .label{ color: var(--agri-muted); font-size:.95rem; }

      /* Feature cards */
      .feature{
        border:1px solid var(--agri-border);
        padding:16px;
        border-radius:18px;
        background:#FFFFFF;
        height:100%;
        box-shadow: 0 3px 16px rgba(0,0,0,.03);
      }
      .feature h4{ margin:.2rem 0 .4rem 0; }

      /* Guidance card */
      .eco-card{
        background: #FFFFFF;
        border:1px solid var(--agri-border);
        border-radius:22px;
        padding:18px 16px;
        margin: 10px 0 18px 0;
        box-shadow: 0 3px 16px rgba(0,0,0,.04);
      }
      .eco-head{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }
      .eco-emoji{ font-size:1.4rem; }
      .eco-title{ font-weight:800; }
      .eco-badge{
        margin-left:auto;
        background: var(--agri-pill);
        color: var(--agri-primary-dark);
        border:1px solid var(--agri-border);
        border-radius:999px;
        padding:4px 10px;
        font-size:.85rem;
      }
      .eco-meta{ margin: 6px 0 8px 0; color: var(--agri-muted); font-size:.95rem; }
      .eco-section-title{ font-weight:800; margin-top:8px; margin-bottom:4px; }
      .eco-list{ margin:0 0 4px 0; padding-left:18px;}
      .eco-list li{ margin: 2px 0; }
      .chip-row{ display:flex; flex-wrap:wrap; gap:8px; margin: 6px 0 2px 0; }
      .chip{
        background: var(--agri-pill);
        color: var(--agri-primary-dark);
        border:1px solid var(--agri-border);
        border-radius:999px;
        padding:4px 10px;
        font-size:.88rem;
      }
      .eco-links{ display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }
      .eco-link{
        border-radius:999px;
        padding:8px 12px;
        border:1px solid var(--agri-border);
        background: #fff;
        text-decoration:none !important;
        color: var(--agri-primary-dark) !important;
        font-weight:700;
      }
      .eco-link:hover{ background: var(--agri-pill); }

      /* Expanders = light FAQ style */
      details{
        background: #FFFFFF;
        border-radius:16px;
        border:1px solid var(--agri-border);
      }
      summary{ padding:8px 10px; }

      /* Subtle hr */
      .soft-hr{ height:1px; background: var(--agri-border); margin: 8px 0 16px 0; }
      .footnote{ text-align:center; color:var(--agri-muted); margin-top:8px; }
    </style>
    """, unsafe_allow_html=True)

apply_agri_theme()

# ======================= Config & Model =======================
MODEL_URL     = os.getenv("MODEL_URL", "https://raw.githubusercontent.com/Bellzum/streamlit-main/main/new_taco1.pt")
LOCAL_MODEL   = os.getenv("LOCAL_MODEL", "best.pt")
CACHED_PATH   = "/tmp/models/best.pt"
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))

CLASS_NAMES   = ["Clear plastic bottle", "Drink can", "Plastic bottle cap"]
IMGSZ_OPTIONS = [320, 416, 512, 640, 800, 960, 1280]

# ======================= Guidance content =======================
SHIBUYA_GUIDE_URL = "https://www.city.shibuya.tokyo.jp/contents/living-in-shibuya/en/daily/garbage.html"

GUIDE = {
    "Clear plastic bottle": {
        "title": "PET bottle (resource)",
        "emoji": "🧴",
        "materials": "Bottle body is PET (polyethylene terephthalate). Caps and labels are usually PP or PE.",
        "why_separate": [
            "Bottles are PET while caps and labels are PP or PE. Mixing reduces bottle-to-bottle quality.",
            "Shibuya asks you to remove caps and labels and sort them with Plastics."
        ],
        "steps": [
            "Remove the cap and label.",
            "Rinse the bottle.",
            "Crush it flat.",
            "Put PET bottles in a transparent bag for PET.",
            "Put caps and labels with Plastics."
        ],
        "recycles_to": ["New PET bottles", "Fibers for clothing and bags", "Sheets, films, and molded goods"],
        "facts": [
            {"text": "Japan turns used PET into new bottles, sheet products, and fibers for clothing.", "url": "https://www.petbottle-rec.gr.jp/english/actual.html"},
            {"text": "Bottle-to-bottle recycling is widely used in Japan.", "url": "https://www.suntory.com/csr/story/003/"}
        ],
        "link": SHIBUYA_GUIDE_URL,
    },
    "Drink can": {
        "title": "Aluminum or steel can (resource)",
        "emoji": "🥫",
        "materials": "Mostly aluminum in Japan. Some cans are steel.",
        "why_separate": [
            "Clean metal cans keep a high-value recycling stream.",
            "Recycling aluminum saves about 95% of the energy compared with smelting new metal."
        ],
        "steps": ["Rinse the can.", "Put cans in a transparent bag for cans."],
        "recycles_to": ["New beverage cans (can-to-can)", "Other aluminum goods such as automotive or construction parts"],
        "facts": [
            {"text": "Aluminum recycling saves about 95% of the energy needed for primary production.", "url": "https://international-aluminium.org/landing/aluminium-recycling-saves-95-of-the-energy-needed-for-primary-aluminium-production/"},
            {"text": "Japan has established can-to-can systems and even 100% recycled aluminum cans on shelves.", "url": "https://www.tskg-hd.com/news_file/file/Toyo%20Seikan%20realizes%20the%20World%27s%20First%20100%20Recycled%20Aluminum%20Beverage%20Can.pdf"}
        ],
        "link": SHIBUYA_GUIDE_URL,
    },
    "Plastic bottle cap": {
        "title": "Plastic bottle cap (plastics)",
        "emoji": "🔘",
        "materials": "PP or PE (polypropylene or polyethylene) closures.",
        "why_separate": [
            "Caps are not PET. Separating avoids contaminating the PET bottle stream.",
            "Shibuya sorts caps with Plastics, not with PET bottles.",
            "Japan is piloting cap-to-cap horizontal recycling."
        ],
        "steps": ["Remove from the bottle.", "If dirty, rinse quickly.", "Put caps with Plastics in a transparent bag.", "Do not put caps with the PET bottle bag."],
        "recycles_to": ["New caps in cap-to-cap pilots", "Plastic containers and packaging", "Pallets and other molded products"],
        "facts": [
            {"text": "Caps and labels go with Plastics in Shibuya, separate from PET bottles.", "url": "https://files.city.shibuya.tokyo.jp/assets/12995aba8b194961be709ba879857f70/bfda2f5d763343b5a0b454087299d57f/2024wakedashiEnglish.pdf"},
            {"text": "Cap-to-cap horizontal recycling is being verified in Japan.", "url": "https://www.sojitz.com/en/news/article/topics-20230112_02.html"}
        ],
        "link": SHIBUYA_GUIDE_URL,
    },
}

def _guide_link(url: str, label: str):
    st.markdown(f'<a class="eco-link" href="{url}" target="_blank" rel="noopener">{label}</a>', unsafe_allow_html=True)

def show_shibuya_guidance(label: str, count: int = 0):
    info = GUIDE.get(label)
    if not info: return
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)
    st.markdown(f"""
      <div class="eco-head">
        <div class="eco-emoji">{info['emoji']}</div>
        <div class="eco-title">Shibuya disposal: {info['title']}</div>
        <div class="eco-badge">Detected: {count}</div>
      </div>
    """, unsafe_allow_html=True)
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
    st.markdown('<div class="eco-links">', unsafe_allow_html=True)
    _guide_link(info["link"], "Official Shibuya guidance")
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

def draw_boxes(bgr, dets):
    import cv2
    out = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (28,160,78), 2)  # green that matches theme
        label = f'{d["class_name"]} {d["score"]:.2f}'
        y = max(y1 - 7, 7)
        cv2.putText(out, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (28,160,78), 1, cv2.LINE_AA)
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

# ======================= UI (Header + Hero + Sections) =======================
# Optional top note
st.markdown('<div class="topnote">Helping Shibuya sort smarter — PET bottles, cans, and caps.</div>', unsafe_allow_html=True)

logo_col, title_col, cta_col = st.columns([1, 6, 2], vertical_alignment="center")
with logo_col:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)  # no deprecated arg
with title_col:
    st.markdown("<div style='font-weight:800; font-size:1.2rem;'>When AI Sees Litter — Shibuya</div>", unsafe_allow_html=True)
with cta_col:
    st.markdown("<div style='text-align:right;'><a class='cta' href='#run'>Try Webcam</a></div>", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <h1>Your Sustainable Sorting <span class="pill">Begins Here</span></h1>
  <p>Detect PET bottles, cans, and caps — then follow Shibuya's guidance and learn what recycling turns them into (clothes, bottles, pallets, and more).</p>
  <div class="cta-row">
    <a class="cta" href="#run">Use Webcam</a>
    <a class="cta secondary" href="#learn">How it works</a>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Model source"):
    st.write(f"LOCAL_MODEL: {LOCAL_MODEL}")
    st.write(f"MODEL_URL: {MODEL_URL or '(empty)'}")
    if os.path.exists(CACHED_PATH):
        st.write(f"Cached path: {CACHED_PATH}  size: {os.path.getsize(CACHED_PATH)/1e6:.2f} MB")

# KPI row
st.markdown('<div class="section">', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown('<div class="kpi"><div class="big">12m t</div><div class="label">Waste avoided</div></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="kpi"><div class="big">25k+</div><div class="label">Predictions</div></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="kpi"><div class="big">73.8%</div><div class="label">Can-to-can (JP)</div></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="kpi"><div class="big">95%</div><div class="label">Energy saved (Al)</div></div>', unsafe_allow_html=True)
st.markdown('<div class="footnote">Demo numbers for illustration</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Feature grid
st.markdown('<div id="learn" class="section">', unsafe_allow_html=True)
f1,f2,f3 = st.columns(3)
with f1: st.markdown('<div class="feature">🌱<h4>Cleaner Streets</h4><p>Instant guidance reduces litter & contamination in public bins.</p></div>', unsafe_allow_html=True)
with f2: st.markdown('<div class="feature">♻️<h4>High-value Recycling</h4><p>Separate PET vs PP/PE caps so bottles can go bottle-to-bottle.</p></div>', unsafe_allow_html=True)
with f3: st.markdown('<div class="feature">📈<h4>Impact Insights</h4><p>Track items guided, diversion rate, and learning engagement.</p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="soft-hr"></div>', unsafe_allow_html=True)

# ======================= Defaults / Advanced settings =======================
# Minimum filters as auto default
_MIN_CONF = 0.05
_MIN_IOU = 0.10
_MIN_IMGSZ = _closest_size(DEFAULT_IMGSZ, IMGSZ_OPTIONS)
_MIN_BOTTLE = 0.00
_MIN_CAN = 0.00
_MIN_CAP = 0.00
_MIN_AREA_PCT = 0.0
_MIN_TTA = False

with st.expander("Advanced settings (optional)"):
    preset = st.radio("Preset", ["Minimum filters", "Recommended", "Strict"], index=0, horizontal=True)

    # Start from Minimum filters
    conf = _MIN_CONF; iou = _MIN_IOU; imgsz = _MIN_IMGSZ
    bottle_min = _MIN_BOTTLE; can_min = _MIN_CAN; cap_min = _MIN_CAP
    min_area_pct = _MIN_AREA_PCT; tta = _MIN_TTA

    if preset == "Recommended":
        conf = 0.25; iou = 0.45
        bottle_min = 0.60; can_min = 0.55; cap_min = 0.65
        min_area_pct = 0.3; tta = False
    elif preset == "Strict":
        conf = 0.35; iou = 0.50
        bottle_min = 0.70; can_min = 0.70; cap_min = 0.75
        min_area_pct = 0.5; tta = False

    conf = st.slider("Base confidence", 0.05, 0.95, conf, 0.01, help="Model confidence threshold.")
    iou  = st.slider("IoU", 0.10, 0.90, iou, 0.01)
    imgsz = st.select_slider("Inference image size", options=IMGSZ_OPTIONS, value=_closest_size(int(imgsz), IMGSZ_OPTIONS))
    c1, c2, c3, c4 = st.columns(4)
    bottle_min = c1.slider("Min conf: Bottle", 0.0, 1.0, bottle_min, 0.01)
    can_min    = c2.slider("Min conf: Can",    0.0, 1.0, can_min, 0.01)
    cap_min    = c3.slider("Min conf: Cap",    0.0, 1.0, cap_min, 0.01)
    min_area_pct = c4.slider("Min box area (%)", 0.0, 5.0, min_area_pct, 0.1, help="Ignore tiny boxes by percent of image area.")
    tta = st.toggle("Test time augmentation", value=tta, help="Slower. Sometimes reduces false positives.")

# If Advanced wasn't opened, fall back to Minimum filters
if "conf" not in locals():
    conf = _MIN_CONF; iou = _MIN_IOU; imgsz = _MIN_IMGSZ
    bottle_min = _MIN_BOTTLE; can_min = _MIN_CAN; cap_min = _MIN_CAP
    min_area_pct = _MIN_AREA_PCT; tta = _MIN_TTA

# ======================= Input & Inference =======================
st.markdown('<div id="run"></div>', unsafe_allow_html=True)
src = st.radio("Input source", ["Upload image", "Webcam"], horizontal=True)
image = None
if src == "Upload image":
    up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if up: image = Image.open(up).convert("RGB")
else:
    shot = st.camera_input("Take a photo", key="cam1")
    if shot: image = Image.open(shot).convert("RGB")

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

                # Filters
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

                # Debug panels (tidy)
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
