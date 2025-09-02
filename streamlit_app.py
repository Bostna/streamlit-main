import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="When AI Sees Litter — Shibuya", page_icon="♻️", layout="wide")

# ================= THEME (Light/Dark) =================
def apply_theme(mode: str):
    # palettes
    if mode == "Eco Dark":
        vars = dict(
            primary="#7EE787", accent="#44BD87",
            bg="#0F1E17", card="#12251C", text="#EAF7EF", muted="#bde7c6",
            pill="#143022", border="rgba(255,255,255,.08)",
            hero_grad_1="#13261E", hero_grad_2="#0F1E17", debug_bg="rgba(20,48,34,.35)"
        )
    else:  # Eco Light (default)
        vars = dict(
            primary="#2E7D32", accent="#43A047",
            bg="#F6FFF8", card="#FFFFFF", text="#0B3D2E", muted="#4F6F5A",
            pill="#E8F5E9", border="#DCEFE3",
            hero_grad_1="#EAF7EF", hero_grad_2="#F6FFF8", debug_bg="rgba(46,125,50,.06)"
        )

    st.markdown(f"""
    <style>
      :root{{
        --eco-primary:{vars['primary']};
        --eco-accent:{vars['accent']};
        --eco-bg:{vars['bg']};
        --eco-card:{vars['card']};
        --eco-text:{vars['text']};
        --eco-muted:{vars['muted']};
        --eco-pill:{vars['pill']};
        --eco-border:{vars['border']};
      }}
      html, body, [data-testid="stAppViewContainer"]{{ background: var(--eco-bg); color: var(--eco-text); }}
      .main .block-container {{ padding-top: 1.0rem !important; }}

      /* Hero */
      .hero{{
        background:
          radial-gradient(900px 360px at 10% -10%, color-mix(in srgb, var(--eco-primary) 20%, transparent), transparent),
          linear-gradient(135deg, {vars['hero_grad_1']} 0%, {vars['hero_grad_2']} 100%);
        border:1px solid var(--eco-border);
        border-radius:18px; padding:22px 20px; margin:8px 0 18px 0;
      }}
      .hero h2{{ margin:0 0 6px 0; font-size:1.6rem; font-weight:800; }}
      .hero p{{ margin:0 0 12px 0; color:var(--eco-muted); }}
      .cta-row{{ display:flex; gap:10px; flex-wrap:wrap }}
      .cta{{
        background:var(--eco-primary); color:var(--eco-bg); padding:10px 14px;
        border-radius:10px; font-weight:800; text-decoration:none !important;
      }}
      .cta.secondary{{
        background:transparent; color:var(--eco-primary);
        border:1px solid color-mix(in srgb, var(--eco-primary) 40%, transparent);
        font-weight:700;
      }}

      /* Sections / KPIs / Features */
      .section{{ margin: 8px 0 22px 0; padding:18px; border:1px solid var(--eco-border); border-radius:16px; background:var(--eco-card); }}
      .kpi{{ text-align:center; border:1px solid var(--eco-border); padding:16px; border-radius:14px; background:color-mix(in srgb, var(--eco-card) 80%, var(--eco-bg) 20%); }}
      .kpi .big{{ font-size:1.6rem; font-weight:800; line-height:1.2; }}
      .kpi .label{{ color:var(--eco-muted); font-size:.95rem; }}
      .feature{{ border:1px solid var(--eco-border); padding:16px; border-radius:14px; background:color-mix(in srgb, var(--eco-card) 80%, var(--eco-bg) 20%); height:100%; }}
      .feature h4{{ margin:.2rem 0 .4rem 0; }}
      .footnote{{ text-align:center; color:var(--eco-muted); margin-top:8px; }}

      /* Guidance card */
      .eco-card{{ background: var(--eco-card); border:1px solid var(--eco-border); border-radius:18px; padding:18px 16px; margin: 10px 0 18px 0; box-shadow: 0 2px 10px rgba(0,0,0,.08); }}
      .eco-head{{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
      .eco-emoji{{ font-size:1.4rem; }}
      .eco-title{{ font-weight:800; }}
      .eco-badge{{ margin-left:auto; background: var(--eco-pill); color: var(--eco-primary); border:1px solid var(--eco-border); border-radius:999px; padding:4px 10px; font-size:.85rem; }}
      .eco-meta{{ margin: 6px 0 8px 0; color: var(--eco-muted); font-size:.95rem; }}
      .eco-section-title{{ font-weight:800; margin-top:8px; margin-bottom:4px; }}
      .eco-list{{ margin:0 0 4px 0; padding-left:18px; }}
      .eco-list li{{ margin: 2px 0; }}
      .chip-row{{ display:flex; flex-wrap:wrap; gap:8px; margin: 6px 0 2px 0; }}
      .chip{{ background: var(--eco-pill); color: var(--eco-primary); border:1px solid var(--eco-border); border-radius:999px; padding:4px 10px; font-size:.88rem; }}
      .eco-links{{ display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }}
      .eco-link{{ border-radius:10px; padding:8px 12px; border:1px solid var(--eco-border); background: transparent; text-decoration:none !important; color:var(--eco-primary) !important; font-weight:700; }}
      .eco-link:hover{{ background: var(--eco-pill); }}

      /* Debug expanders subtle */
      details{{ background: {vars['debug_bg']}; border-radius:12px; border:1px solid var(--eco-border); }}
      summary{{ padding:8px 10px; }}
    </style>
    """, unsafe_allow_html=True)

# Sidebar theme toggle (set BEFORE injecting CSS)
theme_mode = st.sidebar.selectbox("Theme", ["Eco Light", "Eco Dark"], index=0)
apply_theme(theme_mode)

# ------------------------- Config -------------------------
MODEL_URL     = os.getenv("MODEL_URL", "https://raw.githubusercontent.com/Bellzum/streamlit-main/main/new_taco1.pt")
LOCAL_MODEL   = os.getenv("LOCAL_MODEL", "best.pt")
CACHED_PATH   = "/tmp/models/best.pt"
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))

CLASS_NAMES   = ["Clear plastic bottle", "Drink can", "Plastic bottle cap"]
IMGSZ_OPTIONS = [320, 416, 512, 640, 800, 960, 1280]

# --------- Shibuya disposal guidance (materials & uses) ----------
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
    if not info:
        return
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

# ------------------------- Helpers -------------------------
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

# ------------------------- UI (Header + Hero + Sections) -------------------------
logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
with logo_col:
    if os.path.exists("logo.png"):
        # FIX: removed deprecated use_column_width
        st.image("logo.png", width=140)
with title_col:
    st.title("♻️ When AI Sees Litter — Shibuya")

st.markdown("""
<div class="hero">
  <h2>Save the world! <span style="color:var(--eco-primary)">Sustainable</span> litter detection for Shibuya.</h2>
  <p>Detect PET bottles, cans, and caps — then teach correct sorting with official city guidance and
     show what recycling turns them into (clothes, bottles, pallets, and more).</p>
  <div class="cta-row">
    <a class="cta" href="#run">Try Webcam</a>
    <a class="cta secondary" href="#learn">How it works</a>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Model source"):
    st.write(f"LOCAL_MODEL: {LOCAL_MODEL}")
    st.write(f"MODEL_URL: {MODEL_URL or '(empty)'}")
    if os.path.exists(CACHED_PATH):
        st.write(f"Cached path: {CACHED_PATH}  size: {os.path.getsize(CACHED_PATH)/1e6:.2f} MB")

# KPIs
st.markdown('<div class="section">', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown('<div class="kpi"><div class="big">12m t</div><div class="label">Waste avoided</div></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="kpi"><div class="big">25k+</div><div class="label">Predictions</div></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="kpi"><div class="big">73.8%</div><div class="label">Can-to-can (JP)</div></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="kpi"><div class="big">95%</div><div class="label">Energy saved (Al)</div></div>', unsafe_allow_html=True)
st.markdown('<div class="footnote">Demo numbers for illustration</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Features
st.markdown('<div id="learn" class="section">', unsafe_allow_html=True)
f1,f2,f3 = st.columns(3)
with f1: st.markdown('<div class="feature">🌿<h4>Cleaner Streets</h4><p>Cut litter & contamination with instant guidance in Japanese municipalities.</p></div>', unsafe_allow_html=True)
with f2: st.markdown('<div class="feature">♻️<h4>High-value Recycling</h4><p>Separate PET vs PP/PE caps so bottles can go bottle-to-bottle.</p></div>', unsafe_allow_html=True)
with f3: st.markdown('<div class="feature">📊<h4>Impact Insights</h4><p>Track items guided, diversion rate, and learning engagement.</p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')

# ------------------------- Defaults / Advanced settings -------------------------
# Auto default to Minimum filters (even if Advanced isn't opened)
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

# ------------------------- Input & Inference -------------------------
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

                # Debug panels collapsed by default
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
