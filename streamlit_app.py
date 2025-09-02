import os
import shutil
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="When AI Sees Litter — Shibuya", page_icon="♻️", layout="wide")

# ======================= THEME (Light, clean) =======================
def apply_theme():
    st.markdown("""
    <style>
      :root{
        --pri:#79C16D; --pri2:#4FA25A; --hi:#CFEAC0; --bg:#FAFEF6; --card:#FFFFFF;
        --txt:#0F2A1C; --mut:#6F8B7A; --pill:#EEF7E9; --bd:#E5EFE3;
      }
      html, body, [data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--txt); }
      .main .block-container{ padding-top:1rem !important; max-width:1200px; }
      .hero{
        background: radial-gradient(700px 280px at 10% -20%, rgba(121,193,109,.18), transparent),
                    linear-gradient(135deg, #FFFFFF 0%, #F7FBF2 100%);
        border:1px solid var(--bd); border-radius:24px; padding:22px 20px; margin:6px 0 12px 0;
      }
      .hero h1{ margin:0 0 6px 0; font-weight:900; letter-spacing:.2px; font-size:1.8rem; }
      .hero p{ margin:0 0 10px 0; color:var(--mut); }
      .pill{ display:inline-block; background:var(--pill); padding:2px 10px 4px 10px;
             border-radius:999px; color:var(--pri2); border:1px solid var(--bd); }
      .section{ margin:10px 0 18px 0; padding:16px; background:var(--card);
                border:1px solid var(--bd); border-radius:18px; }
      .eco-card{ background:#fff; border:1px solid var(--bd); border-radius:22px; padding:18px 16px;
                 margin:10px 0 18px 0; box-shadow:0 3px 16px rgba(0,0,0,.04); }
      .eco-head{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }
      .eco-emoji{ font-size:1.5rem; }
      .eco-title{ font-weight:900; font-size:1.28rem; }
      .eco-badge{ margin-left:auto; background:var(--pill); color:var(--pri2);
                  border:1px solid var(--bd); border-radius:999px; padding:4px 10px; font-size:.85rem; }
      .eco-meta{ margin:6px 0 8px 0; color:var(--mut); font-size:.95rem; }
      .eco-section-title{ font-weight:800; margin:8px 0 4px 0; }
      .eco-list{ margin:0 0 4px 0; padding-left:18px; }
      .eco-list li{ margin:2px 0; }
      .chip-row{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 2px 0; }
      .chip{ background:var(--pill); color:var(--pri2); border:1px solid var(--bd);
             border-radius:999px; padding:4px 10px; font-size:.88rem; }
      .eco-links{ display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }
      .eco-link{ border-radius:999px; padding:8px 12px; border:1px solid var(--bd);
                 background:#fff; text-decoration:none !important; color:var(--pri2) !important; font-weight:700; }
      .eco-link:hover{ background:var(--pill); }
      .howto li{ margin:2px 0; }
      .sdg-row{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
      .sdg-card{ display:flex; gap:10px; align-items:center; border:1px solid var(--bd);
                 background:#fff; padding:10px 12px; border-radius:14px; }
      .sdg-card img{ width:64px; height:auto; }
      .sdg-card .txt{ font-weight:700; }
      .citybadge{ display:inline-block; background:var(--pill); padding:4px 10px;
                  border-radius:999px; border:1px solid var(--bd); color:var(--pri2); }
    </style>
    """, unsafe_allow_html=True)
apply_theme()

# ======================= Config & Model =======================
# Update your model URL here
MODEL_URL   = os.getenv("MODEL_URL", "https://raw.githubusercontent.com/Bellzum/streamlit-main/main/new_taco1.pt")

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "best.pt")
CACHED_DIR  = "/tmp/models"
# cache file unique to URL so changing MODEL_URL actually refreshes
def _hash_url(u: str) -> str:
    return hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]
CACHED_PATH = os.path.join(CACHED_DIR, f"weights_{_hash_url(MODEL_URL)}.pt")

DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))
IMGSZ_OPTIONS = [320, 416, 512, 640, 800, 960, 1280]

# Force names so class id 2 shows as Plastic bottle cap (UI + thresholds + guidance)
FORCE_CLASS_NAMES = True
TARGET_NAMES = ["Clear plastic bottle", "Drink can", "Plastic bottle cap"]

# ======================= Official references & images =======================
SHIBUYA_GUIDE_URL    = "https://www.city.shibuya.tokyo.jp/contents/living-in-shibuya/en/daily/garbage.html"
SHIBUYA_POSTER_EN    = "https://files.city.shibuya.tokyo.jp/assets/12995aba8b194961be709ba879857f70/bfda2f5d763343b5a0b454087299d57f/2024wakedashiEnglish.pdf#page=2"
SHIBUYA_PLASTICS_NOTICE = "https://files.city.shibuya.tokyo.jp/assets/12995aba8b194961be709ba879857f70/0cdf099fdfe8456fbac12bb5ad7927e4/assets_kusei_ShibuyaCityNews2206_e.pdf#page=1"

# PET steps (illustrative)
FUKUOKA_PET_STEPS = [
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph04.png",
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph05.png",
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph06.png",
    "https://kateigomi-bunbetsu.city.fukuoka.lg.jp/files/Rules/images/bottles/ph07.png",
]

# Recycling marks
ICON_PET   = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Recycling_pet.svg/120px-Recycling_pet.svg.png"
ICON_AL    = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Recycling_alumi.svg/120px-Recycling_alumi.svg.png"
ICON_STEEL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Recycling_steel.svg/120px-Recycling_steel.svg.png"
ICON_PLA   = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Recycling_pla.svg/120px-Recycling_pla.svg.png"

# Carbon-credit info links
LINK_UN_CNP  = "https://unfccc.int/climate-action/united-nations-carbon-offset-platform"
LINK_UN_CNP2 = "https://offset.climateneutralnow.org/"
LINK_WB_MRV  = "https://www.worldbank.org/en/news/feature/2022/07/27/what-you-need-to-know-about-the-measurement-reporting-and-verification-mrv-of-carbon-credits"
LINK_GS      = "https://www.goldstandard.org/"
LINK_VERRA   = "https://verra.org/programs/verified-carbon-standard/"
LINK_JCREDIT = "https://japancredit.go.jp/english/"

# Can-to-can figures
HANWA_CAN2CAN = "https://www.hanwa.co.jp/images/csr/business/img_5_01.png"
CCBJI_CAN2CAN = "https://en.ccbji.co.jp/upload/images/20221222-1-1(5).jpg"

# ======================= Guidance content (Shibuya) =======================
GUIDE_SHIBUYA = {
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
                "If private or rate-limited, make the file public or commit it to this repo."
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
    # Apply forced names in-session so pred.names aligns
    if FORCE_CLASS_NAMES:
        try:
            m.names = {i: n for i, n in enumerate(TARGET_NAMES)}
        except Exception:
            pass
    return m

def load_model():
    path = _ensure_model_path()
    return _load_model_cached(path, _cache_key_for(path))

# ======================= Utils =======================
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return arr[:, :, ::-1]

def draw_boxes(bgr, dets):
    import cv2
    out = bgr.copy()
    H, W = out.shape[:2]
    color = (28,160,78)  # green, BGR
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{d["class_name"]} {d["score"]:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX; fs, thick = 0.5, 1
        (tw, th), _ = cv2.getTextSize(label, font, fs, thick)
        y_text = y1 - 4
        if y_text - th - 4 < 0: y_text = min(y1 + th + 6, H - 2)
        x_text = max(0, min(x1, W - tw - 6))
        cv2.rectangle(out, (x_text, max(0, y_text - th - 4)),
                           (min(x_text + tw + 6, W - 1), min(y_text + 2, H - 1)), color, -1)
        cv2.putText(out, label, (x_text + 3, y_text - 2), font, fs, (255,255,255), 1, cv2.LINE_AA)
    return Image.fromarray(out[:, :, ::-1])

def _get_names_map(pred, model):
    if FORCE_CLASS_NAMES:
        return {i: n for i, n in enumerate(TARGET_NAMES)}
    # fallback to checkpoint names
    if hasattr(pred, "names") and isinstance(pred.names, dict):  return pred.names
    if hasattr(model, "names") and isinstance(model.names, dict): return model.names
    if hasattr(model, "names") and isinstance(model.names, list): return {i:n for i,n in enumerate(model.names)}
    return {0:"Clear plastic bottle", 1:"Drink can", 2:"Plastic bottle cap"}

def _closest_size(target: int, options: list[int]) -> int:
    return min(options, key=lambda x: abs(x - target))

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

def show_guidance_card(label: str, count: int = 0, GUIDE=None):
    info = GUIDE.get(label) if GUIDE else None
    if not info: return
    st.markdown('<div class="eco-card">', unsafe_allow_html=True)
    st.markdown(f"""
      <div class="eco-head">
        <div class="eco-emoji">{info['emoji']}</div>
        <div class="eco-title">{info['title']}</div>
        <div class="eco-badge">Detected: {count}</div>
      </div>
    """, unsafe_allow_html=True)
    if info.get("icons"):
        st.image(info["icons"], width=48, caption=[""]*len(info["icons"]))
    imgs = info.get("images") or []
    if imgs:
        left, right = st.columns([1, 2], vertical_alignment="center")
        with left:
            if len(imgs) == 1:
                st.image(imgs[0], use_container_width=True)
            elif len(imgs) <= 3:
                for im in imgs: st.image(im, use_container_width=True)
            else:
                st.image(imgs, width=160, caption=[""]*len(imgs))
        with right:
            _guidance_text(info)
    else:
        _guidance_text(info)
    st.markdown('<div class="eco-links">', unsafe_allow_html=True)
    if info.get("poster"): _guide_link(info["poster"], "Open local poster")
    _guide_link(info["link"], "Official local guidance (site)")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================= HEADER =======================
logo_col, title_col = st.columns([3, 5], vertical_alignment="center")
with logo_col:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
with title_col:
    st.markdown("<div style='font-weight:800; font-size:1.6rem; line-height:1.2'>When AI Sees Litter — Shibuya</div>", unsafe_allow_html=True)

# ======================= City selector (mock) =======================
st.markdown('<div class="section">', unsafe_allow_html=True)
c1, c2 = st.columns([2, 6], vertical_alignment="center")
with c1:
    city_label = st.selectbox("City / Ward", ["Shibuya (Tokyo)"], index=0)
with c2:
    st.markdown("<div class='citybadge'>More cities coming soon</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

city_id = CITY_MAP[city_label]
GUIDE = GUIDE_BY_CITY.get(city_id, {})

# ======================= Hero =======================
st.markdown(f"""
<div class="hero">
  <h1>Scan litter. Get local sorting guidance.</h1>
  <p><span class="pill">Quick Detect</span> works on PET bottles, drink cans, and plastic bottle caps — <b>{city_label}</b>.</p>
</div>
""", unsafe_allow_html=True)

# ======================= Quick Detect =======================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("#### How to use")
st.markdown("""
<ol class="howto">
  <li>Select <b>Upload image</b> (or open your <b>Camera</b>).</li>
  <li>Tap <b>Run detection</b>.</li>
  <li>Follow the card(s) below for disposal steps — tailored to your selected city.</li>
</ol>
""", unsafe_allow_html=True)

# Presets / defaults
_MIN_CONF=0.05; _MIN_IOU=0.10; _MIN_IMGSZ=_closest_size(DEFAULT_IMGSZ, IMGSZ_OPTIONS)
_MIN_BOTTLE=0.00; _MIN_CAN=0.00; _MIN_CAP=0.00; _MIN_AREA_PCT=0.0; _MIN_TTA=False

with st.expander("Advanced settings (optional)"):
    preset = st.radio("Preset", ["Minimum filters", "Recommended", "Strict"], index=0, horizontal=True)
    conf=_MIN_CONF; iou=_MIN_IOU; imgsz=_MIN_IMGSZ
    bottle_min=_MIN_BOTTLE; can_min=_MIN_CAN; cap_min=_MIN_CAP
    min_area_pct=_MIN_AREA_PCT; tta=_MIN_TTA
    if preset == "Recommended":
        conf=0.25; iou=0.45; bottle_min=0.60; can_min=0.55; cap_min=0.65; min_area_pct=0.3; tta=False
    elif preset == "Strict":
        conf=0.35; iou=0.50; bottle_min=0.70; can_min=0.70; cap_min=0.75; min_area_pct=0.5; tta=False
    conf = st.slider("Base confidence", 0.05, 0.95, conf, 0.01)
    iou  = st.slider("IoU", 0.10, 0.90, iou, 0.01)
    imgsz = st.select_slider("Inference image size", options=IMGSZ_OPTIONS, value=_closest_size(int(imgsz), IMGSZ_OPTIONS))
    c1, c2, c3, c4 = st.columns(4)
    bottle_min = c1.slider("Min conf: Bottle", 0.0, 1.0, bottle_min, 0.01)
    can_min    = c2.slider("Min conf: Can",    0.0, 1.0, can_min,    0.01)
    cap_min    = c3.slider("Min conf: Cap",    0.0, 1.0, cap_min,    0.01)
    min_area_pct = c4.slider("Min box area (%)", 0.0, 5.0, min_area_pct, 0.1,
                             help="Ignore tiny boxes by percent of image area.")
    tta = st.toggle("Test time augmentation", value=tta, help="Slower, sometimes reduces false positives.")

if "conf" not in locals():
    conf=_MIN_CONF; iou=_MIN_IOU; imgsz=_MIN_IMGSZ
    bottle_min=_MIN_BOTTLE; can_min=_MIN_CAN; cap_min=_MIN_CAP
    min_area_pct=_MIN_AREA_PCT; tta=_MIN_TTA

src = st.radio("Input source", ["Upload image", "Camera"], index=0, horizontal=True)
image = None
if src == "Upload image":
    up = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if up: image = Image.open(up).convert("RGB")
else:
    shot = st.camera_input("Open your camera", key="cam1")
    if shot: image = Image.open(shot).convert("RGB")

if st.button("Load model"):
    m = load_model()
    # Show what labels we're using
    if FORCE_CLASS_NAMES:
        ckpt_names = None
        nm = getattr(m, "names", None)
        if isinstance(nm, dict): ckpt_names = list(nm.values())
        elif isinstance(nm, list): ckpt_names = nm
        st.info(f"Labels (forced): {TARGET_NAMES}")
        if ckpt_names and ckpt_names != TARGET_NAMES:
            st.warning(f"Checkpoint labels were {ckpt_names} — overriding to {TARGET_NAMES} for UI/guidance.")
    else:
        names_from_model = getattr(m, "names", None)
        if isinstance(names_from_model, dict):
            st.info(f"Checkpoint labels: {list(names_from_model.values())}")
        elif isinstance(names_from_model, list):
            st.info(f"Checkpoint labels: {names_from_model}")
        else:
            st.info("Using fallback label names.")
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
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            clsi   = pred.boxes.cls.cpu().numpy().astype(int)
            names_map = _get_names_map(pred, model)

            per_class_min = {"Clear plastic bottle": bottle_min,
                             "Drink can": can_min,
                             "Plastic bottle cap": cap_min}
            H, W = bgr.shape[:2]
            min_area = (min_area_pct / 100.0) * (H * W)

            dets, counts = [], {}
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
                area = w * h
                c = int(clsi[i])
                name = names_map.get(c, str(c))
                s = float(scores[i])

                if s < per_class_min.get(name, conf): continue
                if area < min_area: continue

                dets.append({"xyxy":[x1,y1,x2,y2], "class_id":c, "class_name":name, "score":s})
                counts[name] = counts.get(name, 0) + 1

            if not dets:
                st.info("All detections were filtered by thresholds. Try lowering per-class thresholds or min box area.")
            else:
                vis_img = draw_boxes(bgr, dets)
                st.subheader("Detections")
                st.image(vis_img, use_container_width=True)

                with st.expander("Raw detections (debug)", expanded=False):
                    st.dataframe(pd.DataFrame(dets))
                if counts:
                    with st.expander("Counts (debug)", expanded=False):
                        st.bar_chart(pd.Series(counts).sort_values(ascending=False))

                detected_labels = sorted({d["class_name"] for d in dets})
                guide_labels = [lbl for lbl in detected_labels if lbl in GUIDE]
                if guide_labels:
                    st.subheader(f"Disposal instructions — {city_label}")
                    for lbl in guide_labels:
                        show_guidance_card(lbl, counts.get(lbl, 0), GUIDE=GUIDE)
                else:
                    st.caption("No local guidance to show for these detections.")
st.markdown('</div>', unsafe_allow_html=True)

# ======================= Impact & SDGs =======================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("#### Impact & SDGs")
st.markdown("""
- **Carbon credits (what they are):** A carbon credit represents **1 tonne of CO₂-equivalent** reduced or removed. Credits exist only when a **registered project** follows an **approved methodology** and passes **MRV**; they are then **issued on a registry** (e.g., Gold Standard, Verra, or Japan’s J-Credit).  
- **This app does not issue credits.** It helps people sort properly. Educational CO₂e-avoided estimates are okay, but they’re **not credits**.
""")
st.markdown(
    f"""
<div class="eco-links">
  <a class="eco-link" href="{LINK_UN_CNP}"  target="_blank" rel="noopener">UN Carbon Offset Platform</a>
  <a class="eco-link" href="{LINK_UN_CNP2}" target="_blank" rel="noopener">Climate Neutral Now</a>
  <a class="eco-link" href="{LINK_WB_MRV}"  target="_blank" rel="noopener">World Bank: MRV</a>
  <a class="eco-link" href="{LINK_GS}"      target="_blank" rel="noopener">Gold Standard</a>
  <a class="eco-link" href="{LINK_VERRA}"   target="_blank" rel="noopener">Verra VCS</a>
  <a class="eco-link" href="{LINK_JCREDIT}" target="_blank" rel="noopener">Japan J-Credit</a>
</div>
""", unsafe_allow_html=True)

st.markdown("**Our SDGs focus:**")
# Use local images: sdg11.png, sdg12.png, sdg13.png
sdg_cols = st.columns(3)
sdgs = [
    ("sdg12.png", "12 Responsible Consumption & Production"),
    ("sdg11.png", "11 Sustainable Cities & Communities"),
    ("sdg13.png", "13 Climate Action"),
]
for col, (img, caption) in zip(sdg_cols, sdgs):
    with col:
        if os.path.exists(img):
            st.image(img, use_container_width=True, caption=caption)
        else:
            st.markdown(f"*Missing {img}* — {caption}")
st.markdown('</div>', unsafe_allow_html=True)
