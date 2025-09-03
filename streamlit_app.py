# streamlit_app.py
import os
import shutil
import hashlib
import time
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import cv2
from collections import Counter
from urllib.parse import urlparse

st.set_page_config(page_title="When AI Sees Litter — Shibuya", page_icon="♻️", layout="wide")

# ======================= THEME (2025 look) =======================
def apply_theme():
    st.markdown("""
    <style>
      /* ---------- Typography & tokens ---------- */
      :root{
        --font: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
                Roboto, "Helvetica Neue", Arial, "Apple Color Emoji","Segoe UI Emoji";
        --fs-hero: clamp(22px, 2.2vw, 30px);
        --fs-h1:   clamp(18px, 1.8vw, 24px);
        --fs-h2:   clamp(16px, 1.5vw, 20px);
        --fs-body: clamp(14px, 1.1vw, 16px);

        --pri:#4FA25A; --pri-700:#2E7B47; --pri-50:#EEF7E9;
        --bg:#FAFEF6; --card:#FFFFFF; --txt:#0F2A1C; --mut:#6F8B7A; --bd:#E5EFE3;

        --elev-1: 0 3px 16px rgba(0,0,0,.04);
        --elev-2: 0 6px 24px rgba(0,0,0,.08);

        /* Level hues (same hue, different lightness) */
        --lvlH:#175e33; --lvlM:#3f9b60; --lvlL:#87cba1; --lvlVL:#dff1e6;
        --lvlH-fg:#fff; --lvlM-fg:#062016; --lvlL-fg:#062016; --lvlVL-fg:#062016;
      }
      @media (prefers-color-scheme: dark){
        :root{
          --bg:#0e1512; --card:#121b17; --txt:#e9f3ee; --mut:#9fb4aa; --bd:#203229;
          --pri:#77d49a; --pri-700:#3aa368; --pri-50:#132019;
          --elev-1: 0 3px 16px rgba(0,0,0,.4);
          --elev-2: 0 6px 24px rgba(0,0,0,.6);
          --lvlH:#0e3e24; --lvlM:#245f3f; --lvlL:#3b8b62; --lvlVL:#274c3d;
          --lvlH-fg:#eaf8f1; --lvlM-fg:#eaf8f1; --lvlL-fg:#eaf8f1; --lvlVL-fg:#eaf8f1;
        }
      }

      html, body, [data-testid="stAppViewContainer"]{
        background:var(--bg); color:var(--txt); font-family:var(--font);
        -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
      }
      .main .block-container{ padding-top:1rem !important; max-width:1120px; }

      /* Headings */
      h1, .stMarkdown h1 { font-size:var(--fs-hero); font-weight:800; letter-spacing:-0.02em; }
      h2, .stMarkdown h2 { font-size:var(--fs-h1);   font-weight:800; letter-spacing:-0.01em; }
      h3, .stMarkdown h3 { font-size:var(--fs-h2);   font-weight:700; }
      p, li, label, span { font-size:var(--fs-body); }

      /* Pills & links */
      .pill{ display:inline-flex; align-items:center; gap:.35rem; background:var(--pri-50);
             padding:4px 10px; border-radius:999px; color:var(--pri-700); border:1px solid var(--bd); font-weight:700; }
      .eco-links{ display:flex; gap:10px; margin:10px 0 22px; flex-wrap:wrap; }
      .eco-link{ border-radius:999px; padding:8px 12px; border:1px solid var(--bd);
                 background:var(--card); text-decoration:none !important; color:var(--pri-700) !important; font-weight:700;
                 transition:transform .08s ease, background .2s ease, box-shadow .2s ease; }
      .eco-link:hover{ background:var(--pri-50); transform:translateY(-1px); box-shadow:var(--elev-1); }
      .citybadge{ display:inline-flex; background:var(--pri-50); padding:4px 10px; border-radius:999px; border:1px solid var(--bd); color:var(--pri-700); }

      /* Cards */
      .eco-card{ background:var(--card); border:1px solid var(--bd); border-radius:22px; padding:18px 16px;
                 margin:10px 0 18px 0; box-shadow:var(--elev-1); transition:box-shadow .2s ease, transform .06s ease; }
      .eco-card:hover{ box-shadow:var(--elev-2); transform:translateY(-1px); }

      .eco-head{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }
      .eco-emoji{ font-size:1.5rem; }
      .eco-title{ font-weight:900; font-size:1.15rem; }
      .eco-badge{ margin-left:auto; background:var(--pri-50); color:var(--pri-700);
                  border:1px solid var(--bd); border-radius:999px; padding:4px 10px; font-size:.85rem; }

      .eco-section-title-primary{ font-weight:900; font-size:1.05rem; color:var(--pri-700); margin:8px 0 6px; }
      .eco-section-title{ font-weight:800; margin:8px 0 4px; }
      .eco-list{ margin:0 0 6px 0; padding-left:18px; }
      .chip-row{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 2px; }
      .chip{ background:var(--pri-50); color:var(--pri-700); border:1px solid var(--bd); border-radius:999px; padding:4px 10px; font-size:.88rem; }

      /* Level cards */
      .level-card{ border-radius:22px; padding:18px 16px; margin:10px 0 18px 0; box-shadow:var(--elev-1); }
      .lvl-high{    background:var(--lvlH);  color:var(--lvlH-fg); }
      .lvl-mod{     background:var(--lvlM);  color:var(--lvlM-fg); }
      .lvl-low{     background:var(--lvlL);  color:var(--lvlL-fg); }
      .lvl-verylow{ background:var(--lvlVL); color:var(--lvlVL-fg); }
      .level-head{ display:flex; align-items:center; gap:10px; margin-bottom:8px; }
      .level-badge{ margin-left:auto; border-radius:999px; padding:4px 10px; font-weight:700;
                    background:rgba(255,255,255,.16); border:1px solid rgba(0,0,0,.08); }

      /* Clean Streamlit chrome */
      [data-testid="stDivider"], hr, [role="separator"]{ display:none !important; }
      [data-testid="stExpander"] details, [data-testid="stExpander"] summary{ border:none !important; box-shadow:none !important; background:transparent !important; }
      [data-testid="stHorizontalBlock"], [data-testid="stVerticalBlock"]{ border:none !important; box-shadow:none !important; background:transparent !important; }
      [data-testid="stHeader"]{ background:transparent !important; }
      [data-testid="stHeader"] div{ border:none !important; box-shadow:none !important; }

      /* Sticky action bar (mobile-friendly) */
      .sticky-actions{
        position:sticky; bottom:12px; z-index:20; display:flex; gap:8px; justify-content:flex-end;
        padding:8px; backdrop-filter:blur(6px);
        background:color-mix(in oklab, var(--bg) 75%, transparent);
        border-radius:14px; border:1px solid var(--bd);
      }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ======================= Config & Model =======================
MODEL_URL   = os.getenv("MODEL_URL", "https://raw.githubusercontent.com/Bellzum/streamlit-main/main/new_taco1.pt")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "best.pt")

CACHED_DIR  = "/tmp/models"
def _hash_url(u: str) -> str: return hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]
CACHED_PATH = os.path.join(CACHED_DIR, f"weights_{_hash_url(MODEL_URL)}.pt")

IMGSZ_OPTIONS = [200, 320, 416, 512, 640, 800, 960, 1280]
FORCE_CLASS_NAMES = True
TARGET_NAMES = ["Clear plastic bottle", "Drink can", "Styrofoam piece"]

# Official references & images
SHIBUYA_GUIDE_URL    = "https://www.city.shibuya.tokyo.jp/contents/living-in-shibuya/en/daily/garbage.html"
SHIBUYA_POSTER_EN    = "https://files.city.shibuya.tokyo.jp/assets/12995aba8b194961be709ba879857f70/bfda2f5d763343b5a0b454087299d57f/2024wakedashiEnglish.pdf#page=2"
SHIBUYA_PLASTICS_NOTICE = "https://files.city.shibuya.tokyo.jp/assets/12995aba8b194961be709ba879857f70/0cdf099fdfe8456fbac12bb5ad7927e4/assets_kusei_ShibuyaCityNews2206_e.pdf#page=1"
FUKUOKA_PET_STEPS = [
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph04.png",
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph05.png",
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph06.png",
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph07.png",
]
ICON_PET   = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Recycling_pet.svg/120px-Recycling_pet.svg.png"
ICON_AL    = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Recycling_alumi.svg/120px-Recycling_alumi.svg.png"
ICON_STEEL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Recycling_steel.svg/120px-Recycling_steel.svg.png"
ICON_PLA   = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Recycling_pla.svg/120px-Recycling_pla.svg.png"

LINK_UN_CNP  = "https://unfccc.int/climate-action/united-nations-carbon-offset-platform"
LINK_UN_CNP2 = "https://offset.climateneutralnow.org/"
LINK_WB_MRV  = "https://www.worldbank.org/en/news/feature/2022/07/27/what-you-need-to-know-about-the-measurement-reporting-and-verification-mrv-of-carbon-credits"
LINK_GS      = "https://www.goldstandard.org/"
LINK_VERRA   = "https://verra.org/programs/verified-carbon-standard/"
LINK_JCREDIT = "https://japancredit.go.jp/english/"
HANWA_CAN2CAN = "https://www.hanwa.co.jp/images/csr/business/img_5_01.png"

# ======================= Guidance content (Shibuya) =======================
GUIDE_SHIBUYA = {
    "Clear plastic bottle": {
        "title": "Shibuya disposal: PET bottle",
        "emoji": "🧴",
        "materials": "Bottle body is PET. Caps and labels are PP or PE.",
        "why_separate": [
            "Caps and labels contaminate the PET stream if left on.",
            "Shibuya asks you to remove caps and labels and sort them with Plastics."
        ],
        "steps": [
            "Remove the cap and label.",
            "Rinse the bottle.",
            "Crush it flat.",
            "Put PET bottles in a transparent bag for PET.",
            "Put caps and labels with Plastics."
        ],
        "recycles_to": ["New PET bottles", "Fibers for clothing and bags", "Sheets and films"],
        "facts": [
            {"text": "Japan’s reported plastic recycling rate includes thermal recovery. Clean PET enables high value bottle to bottle.",
             "url": "https://japan-forward.com/japans-plastic-recycling-the-unseen-reality/"},
        ],
        "images": FUKUOKA_PET_STEPS,
        "icons": [ICON_PET],
        "link": SHIBUYA_GUIDE_URL,
        "poster": SHIBUYA_POSTER_EN,
    },
    "Drink can": {
        "title": "Shibuya disposal: Aluminum or steel can",
        "emoji": "🥫",
        "materials": None,
        "why_separate": [
            "Clean cans keep a high value recycling stream.",
            "Aluminum recycling saves major energy compared with producing new metal."
        ],
        "steps": [
            "Rinse the can.",
            "Optional: Lightly crush or squeeze to save space if your building allows.",
            "Put cans in a transparent bag for cans."
        ],
        "recycles_to": ["New beverage cans", "Automotive & construction parts", "Remelt scrap ingots"],
        "facts": [
            {"text": "Coca-Cola Bottlers Japan promotes CAN-to-CAN (recycled aluminum bodies).",
             "url": "https://en.ccbji.co.jp/news/detail.php?id=1347"},
            {"text": "Hanwa: used aluminum cans are cleaned, melted and supplied as remelt scrap ingots — then used again as cans.",
             "url": HANWA_CAN2CAN},
        ],
        "images": [HANWA_CAN2CAN],
        "icons": [ICON_AL, ICON_STEEL],
        "link": SHIBUYA_GUIDE_URL,
        "poster": SHIBUYA_POSTER_EN,
    },
    "Styrofoam piece": {
        "title": "Shibuya disposal: Styrofoam piece",
        "emoji": "🧊",
        "materials": "Expanded polystyrene foam.",
        "why_separate": [
            "Clean pieces can go to Plastic items when marked as packaging.",
            "Keeping plastics clean improves material recovery quality."
        ],
        "steps": [
            "Remove food residue and wipe or quick rinse if needed.",
            "Break large pieces down to fit bags.",
            "Put Styrofoam with Plastic items in a clear or semi-clear bag following building day."
        ],
        "recycles_to": ["Foam trays & molded parts", "Pellets for plastic goods", "Sometimes thermal recovery"],
        "facts": [
            {"text": "Plastic sorting rules vary by municipality. See Shibuya’s plastics notice for details.",
             "url": SHIBUYA_PLASTICS_NOTICE},
        ],
        "images": ["https://www.fpco.jp/dcms_media/image/appeal_img01_b.jpg"],
        "icons": [ICON_PLA],
        "link": SHIBUYA_GUIDE_URL,
        "poster": SHIBUYA_PLASTICS_NOTICE,
    },
}
GUIDE_BY_CITY = {"shibuya": GUIDE_SHIBUYA}
CITY_MAP = {"Shibuya (Tokyo)": "shibuya"}

# ======================= Download / load model =======================
def _download_file(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for ch in r.iter_content(chunk_size=8192):
                    if ch: f.write(ch)
    except Exception as e_req:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
        except Exception as e_url:
            st.error(
                f"Failed to download model from URL:\n{url}\n\n"
                f"requests error: {e_req}\nurllib error: {e_url}\n\n"
                "If private or rate limited, make the file public or commit it to this repo."
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
    m = YOLO(path)
    if FORCE_CLASS_NAMES:
        try: m.names = {i: n for i, n in enumerate(TARGET_NAMES)}
        except Exception: pass
    return m

def load_model():
    path = _ensure_model_path()
    return _load_model_cached(path, _cache_key_for(path))

GLOBAL_MODEL = load_model()

# ======================= Levels & helpers =======================
def level_for_score(score: float) -> str:
    s = float(score)
    if s >= 0.80: return "High"
    if s >= 0.60: return "Moderate"
    if s >= 0.40: return "Low"
    return "Very Low"

def level_css_class(level: str) -> str:
    return {"High":"lvl-high","Moderate":"lvl-mod","Low":"lvl-low","Very Low":"lvl-verylow"}.get(level,"lvl-low")

def level_box_bgr(level: str) -> tuple[int,int,int]:
    return {"High":(35,110,65),"Moderate":(70,150,100),"Low":(130,190,150),"Very Low":(195,225,205)}[level]

def _domain_label(url: str) -> str:
    try: return urlparse(url).netloc.replace("www.","")
    except Exception: return "link"

def _guide_link(url: str, label: str, tooltip: str | None = None, icon: str | None = None):
    title_attr = f' title="{tooltip}"' if tooltip else ""
    icon_html = f"{icon} " if icon else ""
    st.markdown(
        f'<a class="eco-link" href="{url}" target="_blank" rel="noopener"{title_attr}>{icon_html}{label}</a>',
        unsafe_allow_html=True
    )

def _guidance_text(info: dict):
    st.markdown('<div class="eco-section-title-primary">How to put out</div>', unsafe_allow_html=True)
    st.markdown('<ul class="eco-list">', unsafe_allow_html=True)
    for step in info["steps"]:
        st.markdown(f'<li>{step}</li>', unsafe_allow_html=True)
    st.markdown('</ul>', unsafe_allow_html=True)

    if info.get("why_separate"):
        st.markdown('<div class="eco-section-title">How to manage</div>', unsafe_allow_html=True)
        st.markdown('<ul class="eco-list">', unsafe_allow_html=True)
        for reason in info["why_separate"]:
            st.markdown(f'<li>{reason}</li>', unsafe_allow_html=True)
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
            dom = _domain_label(fact["url"])
            _guide_link(fact["url"], f"🔗 Learn more · {dom}", tooltip=fact["url"])
        st.markdown('</div>', unsafe_allow_html=True)

def show_guidance_card(label: str, count: int = 0, GUIDE=None, level: str | None = None):
    info = GUIDE.get(label) if GUIDE else None
    if not info: return
    css_class = level_css_class(level) if level else "lvl-low"
    st.markdown(f"<div class='level-card {css_class}'>", unsafe_allow_html=True)

    st.markdown(f"""
      <div class="level-head">
        <div class="eco-emoji">{info['emoji']}</div>
        <div class="eco-title">{info['title']}</div>
        <div class="level-badge">{('Level: ' + level) if level else 'Detected'} · {count}</div>
      </div>
    """, unsafe_allow_html=True)

    if info.get("icons"):
        st.image(info["icons"], width=44, caption=[""]*len(info["icons"]))

    imgs = info.get("images") or []
    if imgs:
        left, right = st.columns([1, 2])
        with left:
            if len(imgs) == 1:
                st.image(imgs[0], use_container_width=True)
            elif len(imgs) <= 3:
                for im in imgs: st.image(im, use_container_width=True)
            else:
                st.image(imgs, width=150, caption=[""]*len(imgs))
        with right:
            _guidance_text(info)
    else:
        _guidance_text(info)

    st.markdown('<div class="eco-links">', unsafe_allow_html=True)
    if info.get("poster"):
        dom = _domain_label(info["poster"])
        _guide_link(info["poster"], f"📄 Poster (PDF) · {dom}", tooltip="Open local poster (PDF)")
    if info.get("link"):
        dom = _domain_label(info["link"])
        _guide_link(info["link"], f"🌐 Official guidance · {dom}", tooltip="Official local guidance site")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================= Header =======================
header_col, _ = st.columns([3,5])
with header_col:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)

st.markdown("# When AI Sees Litter — Shibuya")
st.caption("Detect common items and see exactly how to put them out — tuned for Shibuya rules.")

# City/Ward block
c1, c2 = st.columns([2,6])
with c1:
    city_label = st.selectbox("City / Ward", ["Shibuya (Tokyo)"], index=0)
with c2:
    st.markdown("<div class='citybadge'>More cities coming soon</div>", unsafe_allow_html=True)
city_id = CITY_MAP[city_label]
GUIDE = GUIDE_BY_CITY.get(city_id, {})

# ======================= Two-pane layout =======================
left, right = st.columns([6,5], gap="large")

# ----- LEFT: Input & Controls -----
with left:
    st.markdown("### Choose input")
    src = st.radio("Source", ["Upload image", "Camera"], index=0, horizontal=True)
    image = None
    if src == "Upload image":
        up = st.file_uploader("Upload a photo", type=["jpg","jpeg","png"])
        if up: image = Image.open(up).convert("RGB")
    else:
        shot = st.camera_input("Take a photo", key="cam1")
        if shot: image = Image.open(shot).convert("RGB")

    st.markdown("### Options")

    # --- Recommended defaults ---
    _REC_CONF=0.00; _REC_IOU=0.00; _REC_IMGSZ=200
    _REC_BOTTLE=0.20; _REC_CAN=0.20; _REC_FOAM=0.20; _REC_AREA_PCT=0.20; _REC_TTA=True

    # Session state keys for controls
    for k, v in {
        "conf":_REC_CONF, "iou":_REC_IOU, "imgsz":_REC_IMGSZ,
        "bottle_min":_REC_BOTTLE, "can_min":_REC_CAN, "foam_min":_REC_FOAM,
        "min_area_pct":_REC_AREA_PCT, "tta":_REC_TTA,
        "preview_brightness":0, "preview_contrast":1.0, "preview_gamma":1.0,
        "auto_run": True
    }.items():
        st.session_state.setdefault(k, v)

    st.session_state.auto_run = st.toggle("Auto-detect on upload", value=st.session_state.auto_run)

    with st.expander("Advanced settings", expanded=False):
        preset = st.radio("Preset", ["Minimum filters", "Recommended", "Strict"], index=1, horizontal=True)
        cA, cB = st.columns(2)
        with cA:
            st.session_state.conf = st.slider("Model confidence", 0.0, 0.95, float(st.session_state.conf), 0.01,
                                              help="Filter out low-confidence boxes.")
            st.session_state.iou  = st.slider("IoU (NMS)", 0.0, 0.90, float(st.session_state.iou), 0.01,
                                              help="Merge overlapping boxes.")
            st.session_state.imgsz = int(st.select_slider("Inference image size", options=IMGSZ_OPTIONS,
                                                          value=int(st.session_state.imgsz),
                                                          help="Bigger = slower but sharper."))
        with cB:
            st.session_state.bottle_min   = st.slider("Min conf: PET bottle", 0.0, 1.0, float(st.session_state.bottle_min), 0.01)
            st.session_state.can_min      = st.slider("Min conf: Drink can",  0.0, 1.0, float(st.session_state.can_min),    0.01)
            st.session_state.foam_min     = st.slider("Min conf: Styrofoam",  0.0, 1.0, float(st.session_state.foam_min),   0.01)
            st.session_state.min_area_pct = st.slider("Ignore tiny boxes (%)", 0.0, 5.0, float(st.session_state.min_area_pct), 0.1)

        st.session_state.tta = st.toggle("Test-time augmentation", value=st.session_state.tta,
                                         help="Slower. Sometimes reduces false positives.")

        # Preview tone controls (display only; does not affect inference)
        toneA, toneB, toneC = st.columns(3)
        st.session_state.preview_brightness = toneA.slider("Preview brightness", -40, 40, int(st.session_state.preview_brightness))
        st.session_state.preview_contrast   = toneB.slider("Preview contrast", 0.6, 1.6, float(st.session_state.preview_contrast), 0.01)
        st.session_state.preview_gamma      = toneC.slider("Preview gamma", 0.8, 1.4, float(st.session_state.preview_gamma), 0.01)

        def _apply_preset(name: str):
            if name == "Minimum filters":
                vals = dict(conf=0.05, iou=0.10, imgsz=IMGSZ_OPTIONS[0],
                            bottle_min=0.00, can_min=0.00, foam_min=0.00, min_area_pct=0.0, tta=False)
            elif name == "Strict":
                vals = dict(conf=0.35, iou=0.50, imgsz=640,
                            bottle_min=0.70, can_min=0.70, foam_min=0.75, min_area_pct=0.5, tta=False)
            else:  # Recommended
                vals = dict(conf=_REC_CONF, iou=_REC_IOU, imgsz=_REC_IMGSZ,
                            bottle_min=_REC_BOTTLE, can_min=_REC_CAN, foam_min=_REC_FOAM,
                            min_area_pct=_REC_AREA_PCT, tta=_REC_TTA)
            for k, v in vals.items(): st.session_state[k] = v

        cBtns1, cBtns2 = st.columns([1,1])
        if cBtns1.button("Apply preset"):
            _apply_preset(preset); st.rerun()
        if cBtns2.button("Reset to Recommended", type="secondary"):
            _apply_preset("Recommended"); st.rerun()

    # Preview (input)
    if image is not None:
        st.image(image, caption="Input", use_container_width=True)
    else:
        st.info("Upload a photo or take one with your camera.")

    # Sticky run button (useful on mobile)
    st.markdown('<div class="sticky-actions">', unsafe_allow_html=True)
    run_btn = st.button("Run detection", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# ----- RIGHT: Results & Guidance -----
with right:
    st.caption("Model loaded ✅")

    def render_summary(counts: dict):
        a,b,c = st.columns(3)
        a.metric("PET bottles", counts.get("Clear plastic bottle",0))
        b.metric("Drink cans", counts.get("Drink can",0))
        c.metric("Styrofoam pieces", counts.get("Styrofoam piece",0))

# ======================= Core inference helpers =======================
def _get_names_map(pred, model):
    if FORCE_CLASS_NAMES: return {i: n for i, n in enumerate(TARGET_NAMES)}
    if hasattr(pred, "names") and isinstance(pred.names, dict):  return pred.names
    if hasattr(model, "names") and isinstance(model.names, dict): return model.names
    if hasattr(model, "names") and isinstance(model.names, list): return {i:n for i,n in enumerate(model.names)}
    return {0:"Clear plastic bottle", 1:"Drink can", 2:"Styrofoam piece"}

def _apply_preview_tone(img_bgr, brightness:int, contrast:float, gamma:float):
    out = cv2.convertScaleAbs(img_bgr, alpha=contrast, beta=brightness)
    if abs(gamma - 1.0) > 1e-3:
        table = ((np.arange(256)/255.0)**(1.0/gamma)*255).astype("uint8")
        out = cv2.LUT(out, table)
    return out

def run_detection(image_pil: Image.Image):
    model = GLOBAL_MODEL
    # Use original pixels for the model (no tone edits)
    bgr = np.array(image_pil.convert("RGB"))[:, :, ::-1]
    results = model.predict(
        bgr,
        conf=st.session_state.conf,
        iou=st.session_state.iou,
        imgsz=st.session_state.imgsz,
        verbose=False,
        augment=st.session_state.tta
    )
    pred = results[0]
    if pred.boxes is None or len(pred.boxes) == 0:
        return [], {}
    boxes = pred.boxes.xyxy.cpu().numpy()
    scores = pred.boxes.conf.cpu().numpy()
    clsi   = pred.boxes.cls.cpu().numpy().astype(int)

    names_map = {i:n for i,n in enumerate(TARGET_NAMES)} if FORCE_CLASS_NAMES else _get_names_map(pred, model)
    per_class_min = {
        "Clear plastic bottle": st.session_state.bottle_min,
        "Drink can":            st.session_state.can_min,
        "Styrofoam piece":      st.session_state.foam_min,
    }

    H, W = bgr.shape[:2]
    min_area = (st.session_state.min_area_pct / 100.0) * (H * W)

    dets, counts = [], {}
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
        area = w * h
        c = int(clsi[i])
        name = names_map.get(c, str(c))
        s = float(scores[i])
        if s < per_class_min.get(name, st.session_state.conf): continue
        if area < min_area: continue
        dets.append({"xyxy":[x1,y1,x2,y2], "class_id":c, "class_name":name, "score":s})
        counts[name] = counts.get(name, 0) + 1
    return dets, counts

def draw_and_show(image_pil: Image.Image, dets):
    bgr = np.array(image_pil.convert("RGB"))[:, :, ::-1]
    out = bgr.copy()
    H, W = out.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        lvl = level_for_score(float(d["score"]))
        color = level_box_bgr(lvl)  # darker for high, lighter for low
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{d["class_name"]} {d["score"]:.2f} · {lvl}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = y1 - 4
        if y_text - th - 4 < 0: y_text = min(y1 + th + 6, H - 2)
        x_text = max(0, min(x1, W - tw - 6))
        cv2.rectangle(out, (x_text, max(0, y_text - th - 4)),
                           (min(x_text + tw + 6, W - 1), min(y_text + 2, H - 1)), color, -1)
        cv2.putText(out, label, (x_text + 3, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # Apply tone edits only to the DISPLAY (not the model input)
    out = _apply_preview_tone(
        out,
        brightness=int(st.session_state.preview_brightness),
        contrast=float(st.session_state.preview_contrast),
        gamma=float(st.session_state.preview_gamma),
    )
    st.image(Image.fromarray(out[:, :, ::-1]), caption="Detections", use_container_width=True)

# ======================= Orchestration =======================
should_run = False
if 'auto_run' in st.session_state and st.session_state.auto_run:
    # will run later only if image is present
    should_run = True
# User-pressed button overrides auto flag
if 'run_btn' in locals() and run_btn:
    should_run = True

if image is not None and should_run:
    with st.status("Detecting…", expanded=False) as status:
        dets, counts = run_detection(image)
        if dets:
            draw_and_show(image, dets)
            status.update(label="Detected ✅", state="complete", expanded=False)
            with right:
                render_summary(counts)
                # per-label average score → level
                avg_score = {}
                for d in dets:
                    avg_score.setdefault(d["class_name"], []).append(float(d["score"]))
                avg_score = {k: float(np.mean(v)) for k, v in avg_score.items()}

                detected_labels = sorted(avg_score.keys())
                guide_labels = [lbl for lbl in detected_labels if lbl in GUIDE]
                if guide_labels:
                    st.subheader(f"Disposal instructions · {city_label}")
                    for lbl in guide_labels:
                        lvl = level_for_score(avg_score[lbl])
                        show_guidance_card(lbl, counts.get(lbl, 0), GUIDE=GUIDE, level=lvl)
                else:
                    st.caption("No local guidance to show for these detections.")
        else:
            status.update(label="No detections", state="error", expanded=True)
            with right:
                st.info("I didn’t find anything yet. Try more light or move closer (≈⅓ of frame), or lower the thresholds in Advanced settings.")

# ======================= Impact & SDGs =======================
st.markdown("#### Impact & SDGs")
st.markdown("""
- **Carbon credits (what they are):** A carbon credit represents **1 tonne of CO₂-eq** reduced or removed. Credits exist only when a **registered project** follows an **approved methodology** and passes **MRV**; they are then **issued on a registry** (Gold Standard, Verra, Japan’s J-Credit).
- **This app does not issue credits.** It helps you sort properly. Educational CO₂e-avoided estimates are okay, but they’re **not credits**.
""", unsafe_allow_html=True)
st.markdown(
    f"""
<div class="eco-links">
  <a class="eco-link" href="{LINK_UN_CNP}"  target="_blank" rel="noopener" title="United Nations Carbon Offset Platform">UN Carbon Offset Platform</a>
  <a class="eco-link" href="{LINK_UN_CNP2}" target="_blank" rel="noopener" title="UN Climate Neutral Now">Climate Neutral Now</a>
  <a class="eco-link" href="{LINK_WB_MRV}"  target="_blank" rel="noopener" title="World Bank: MRV overview">World Bank: MRV</a>
  <a class="eco-link" href="{LINK_GS}"      target="_blank" rel="noopener" title="Gold Standard registry">Gold Standard</a>
  <a class="eco-link" href="{LINK_VERRA}"   target="_blank" rel="noopener" title="Verra: Verified Carbon Standard">Verra VCS</a>
  <a class="eco-link" href="{LINK_JCREDIT}" target="_blank" rel="noopener" title="Japan's J-Credit program">Japan J Credit</a>
</div>
""", unsafe_allow_html=True)

# SDG tiles
col1, col2, col3 = st.columns(3)
def sdg_tile(col, path, label):
    with col:
        if os.path.exists(path):
            st.image(path, width=180)
        else:
            st.warning(f"Missing {path}")
        st.markdown(f"<div class='sdg-caption'>{label}</div>", unsafe_allow_html=True)

sdg_tile(col1, "sdg12.png", "12 Responsible Consumption & Production")
sdg_tile(col2, "sdg11.png", "11 Sustainable Cities & Communities")
sdg_tile(col3, "sdg13.png", "13 Climate Action")
