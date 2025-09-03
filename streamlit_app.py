import os
import shutil
import hashlib
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="When AI Sees Litter — Shibuya", page_icon="♻️", layout="wide")

# ======================= THEME (light; no section borders/lines) =======================
def apply_theme():
    st.markdown("""
    <style>
      :root{
        --pri:#79C16D; --pri2:#4FA25A; --hi:#CFEAC0; --bg:#FAFEF6; --card:#FFFFFF;
        --txt:#0F2A1C; --mut:#6F8B7A; --pill:#EEF7E9; --bd:#E5EFE3;
      }
      html, body, [data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--txt); }
      .main .block-container{ padding-top:1rem !important; max-width:1200px; }

      /* Headings & chips */
      .pill{ display:inline-block; background:var(--pill); padding:2px 10px 4px 10px;
             border-radius:999px; color:var(--pri2); border:1px solid var(--bd); }

      /* Guidance cards */
      .eco-card{ background:#fff; border:none; border-radius:22px; padding:18px 16px;
                 margin:10px 0 18px 0; box-shadow:0 3px 16px rgba(0,0,0,.04); }
      .eco-head{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }
      .eco-emoji{ font-size:1.5rem; }
      .eco-title{ font-weight:900; font-size:1.28rem; }
      .eco-badge{ margin-left:auto; background:var(--pill); color:var(--pri2);
                  border:1px solid var(--bd); border-radius:999px; padding:4px 10px; font-size:.85rem; }
      .eco-meta{ margin:6px 0 8px 0; color:var(--mut); font-size:.95rem; }

      /* Make "How to put out" stand out */
      .eco-section-title-primary{ font-weight:900; font-size:1.12rem; color:var(--pri2); margin:8px 0 6px 0; }
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

      .citybadge{ display:inline-block; background:var(--pill); padding:4px 10px;
                  border-radius:999px; border:1px solid var(--bd); color:var(--pri2); }

      /* SDGs row (local images @ 180px) */
      .sdg-row{ display:flex; gap:16px; flex-wrap:wrap; align-items:center; }
      .sdg-card{ display:flex; gap:10px; align-items:center; border:none;
                 background:#fff; padding:8px 10px; border-radius:14px; box-shadow:0 2px 10px rgba(0,0,0,.04); }
      .sdg-card img{ width:180px; height:auto; }
      .sdg-card .txt{ font-weight:700; }

      /* Remove all separators / default hr lines / expander borders */
      [data-testid="stDivider"], hr, [role="separator"]{ display:none !important; }
      [data-testid="stExpander"] details, [data-testid="stExpander"] summary{
        border:none !important; box-shadow:none !important; background:transparent !important;
      }
      [data-testid="stHorizontalBlock"], [data-testid="stVerticalBlock"]{
        border:none !important; box-shadow:none !important; background:transparent !important;
      }
      [data-testid="stHeader"]{ background:transparent !important; }
      [data-testid="stHeader"] div{ border:none !important; box-shadow:none !important; }
    </style>
    """, unsafe_allow_html=True)
apply_theme()

# ======================= Config & Model =======================
MODEL_URL   = os.getenv("MODEL_URL", "https://raw.githubusercontent.com/Bellzum/streamlit-main/main/new_taco1.pt")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "best.pt")

CACHED_DIR  = "/tmp/models"
def _hash_url(u: str) -> str: return hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]
CACHED_PATH = os.path.join(CACHED_DIR, f"weights_{_hash_url(MODEL_URL)}.pt")

DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))
IMGSZ_OPTIONS = [320, 416, 512, 640, 800, 960, 1280]

# Force UI names — class 2 is "Styrofoam piece"
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

# Links
LINK_UN_CNP  = "https://unfccc.int/climate-action/united-nations-carbon-offset-platform"
LINK_UN_CNP2 = "https://offset.climateneutralnow.org/"
LINK_WB_MRV  = "https://www.worldbank.org/en/news/feature/2022/07/27/what-you-need-to-know-about-the-measurement-reporting-and-verification-mrv-of-carbon-credits"
LINK_GS      = "https://www.goldstandard.org/"
LINK_VERRA   = "https://verra.org/programs/verified-carbon-standard/"
LINK_JCREDIT = "https://japancredit.go.jp/english/"
HANWA_CAN2CAN = "https://www.hanwa.co.jp/images/csr/business/img_5_01.png"
CCBJI_CAN2CAN = "https://en.ccbji.co.jp/upload/images/20221222-1-1(5).jpg"

# ======================= Guidance content (Shibuya) =======================
GUIDE_SHIBUYA = {
    "Clear plastic bottle": {
        "title": "Shibuya disposal: PET bottle",
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
            }
        ],
        "images": FUKUOKA_PET_STEPS,
        "icons": [ICON_PET],
        "link": SHIBUYA_GUIDE_URL,
        "poster": SHIBUYA_POSTER_EN,
    },
    "Drink can": {
        "title": "Shibuya disposal: Aluminum or steel can",
        "emoji": "🥫",
        "materials": None,  # removed per request
        "why_separate": [
            "Clean cans keep a high-value recycling stream.",
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
                "text": "Hanwa: used aluminum cans are cleaned, melted and supplied as remelt scrap ingots — then used again as cans.",
                "url": HANWA_CAN2CAN
            }
        ],
        "images": [HANWA_CAN2CAN, CCBJI_CAN2CAN],
        "icons": [ICON_AL, ICON_STEEL],
        "link": SHIBUYA_GUIDE_URL,
        "poster": SHIBUYA_POSTER_EN,
    },
    "Styrofoam piece": {
        "title": "Shibuya disposal: Styrofoam piece",
        "emoji": "🧊",
        "materials": "Expanded polystyrene (EPS) foam.",
        "why_separate": [
            "Styrofoam is polystyrene. Clean pieces can go to Plastic items when marked as packaging.",
            "Keeping plastics clean improves material recovery quality."
        ],
        "steps": [
            "Remove food residue; wipe or quick rinse if necessary.",
            "Break large pieces down to fit bags.",
            "Put Styrofoam with Plastic items in a clear/semi-clear bag (follow building day)."
        ],
        "recycles_to": ["Foam trays & molded parts", "Pellets for plastic goods", "(Sometimes) thermal recovery"],
        "facts": [
            {
                "text": "Plastic sorting rules vary by municipality; see Shibuya’s plastics notice for details.",
                "url": SHIBUYA_PLASTICS_NOTICE
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
    if FORCE_CLASS_NAMES:
        try: m.names = {i: n for i, n in enumerate(TARGET_NAMES)}
        except Exception: pass
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
    color = (28,160,78)
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
    if hasattr(pred, "names") and isinstance(pred.names, dict):  return pred.names
    if hasattr(model, "names") and isinstance(model.names, dict): return model.names
    if hasattr(model, "names") and isinstance(model.names, list): return {i:n for i,n in enumerate(model.names)}
    return {0:"Clear plastic bottle", 1:"Drink can", 2:"Styrofoam piece"}

def _guide_link(url: str, label: str):
    st.markdown(f'<a class="eco-link" href="{url}" target="_blank" rel="noopener">{label}</a>', unsafe_allow_html=True)

def _guidance_text(info: dict):
    # ORDER: How to put out (primary) FIRST, then Why separate, then Recycles to, then facts
    st.markdown('<div class="eco-section-title-primary">How to put out</div>', unsafe_allow_html=True)
    st.markdown('<ul class="eco-list">', unsafe_allow_html=True)
    for step in info["steps"]:
        st.markdown(f'<li>{step}</li>', unsafe_allow_html=True)
    st.markdown('</ul>', unsafe_allow_html=True)

    if info.get("why_separate"):
        st.markdown('<div class="eco-section-title">Why separate?</div>', unsafe_allow_html=True)
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
        for fact in facts: _guide_link(fact["url"], "Learn more")
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

# ======================= HEADER (logo only) =======================
logo_col, _ = st.columns([3, 5], vertical_alignment="center")
with logo_col:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)

# ======================= MAIN INTRO =======================
st.markdown("### Let’s Start Sorting!")

# (Move City/Ward block here — between heading and step 1)
c1, c2 = st.columns([2, 6], vertical_alignment="center")
with c1:
    city_label = st.selectbox("City / Ward", ["Shibuya (Tokyo)"], index=0)
with c2:
    st.markdown("<div class='citybadge'>More cities coming soon</div>", unsafe_allow_html=True)
city_id = CITY_MAP["Shibuya (Tokyo)"]
GUIDE = GUIDE_BY_CITY.get(city_id, {})

# Steps (with updated step 3 wording)
st.markdown("""
<ol class="howto">
  <li>**Select Upload image** (or open your **Camera**).</li>
  <li>**Detection runs** and shows results.</li>
  <li>Follow the **custom disposal instructions below for your city**.</li>
</ol>
""", unsafe_allow_html=True)

# ======================= Inputs (source + auto-run + image pickers) =======================
src = st.radio("Input source", ["Upload image", "Camera"], index=0, horizontal=True)
auto_run = st.toggle("Auto-run detection", value=True, help="Run detection automatically after you choose/take a photo.")
image = None

if src == "Upload image":
    up = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if up: image = Image.open(up).convert("RGB")
else:
    shot = st.camera_input("Open your camera", key="cam1")
    if shot: image = Image.open(shot).convert("RGB")

# ======================= Advanced settings (moved below inputs) =======================
# Recommended defaults (auto-applied)
_REC_CONF=0.00; _REC_IOU=0.00; _REC_IMGSZ=640
_REC_BOTTLE=0.20; _REC_CAN=0.20; _REC_FOAM=0.20; _REC_AREA_PCT=0.20; _REC_TTA=False

conf=_REC_CONF; iou=_REC_IOU; imgsz=_REC_IMGSZ
bottle_min=_REC_BOTTLE; can_min=_REC_CAN; foam_min=_REC_FOAM
min_area_pct=_REC_AREA_PCT; tta=_REC_TTA

with st.expander("Advanced settings (optional)"):
    preset = st.radio("Preset", ["Minimum filters", "Recommended", "Strict"], index=1, horizontal=True)
    if preset == "Minimum filters":
        conf=0.05; iou=0.10; imgsz=640
        bottle_min=0.00; can_min=0.00; foam_min=0.00; min_area_pct=0.0; tta=False
    elif preset == "Recommended":
        conf=_REC_CONF; iou=_REC_IOU; imgsz=_REC_IMGSZ
        bottle_min=_REC_BOTTLE; can_min=_REC_CAN; foam_min=_REC_FOAM; min_area_pct=_REC_AREA_PCT; tta=_REC_TTA
    elif preset == "Strict":
        conf=0.35; iou=0.50; imgsz=640
        bottle_min=0.70; can_min=0.70; foam_min=0.75; min_area_pct=0.5; tta=False

    conf = st.slider("Base confidence", 0.0, 0.95, float(conf), 0.01)
    iou  = st.slider("IoU",            0.0, 0.90, float(iou),  0.01)
    imgsz = int(st.select_slider("Inference image size", options=[320,416,512,640,800,960,1280], value=int(imgsz)))
    c1a, c2a, c3a, c4a = st.columns(4)
    bottle_min   = c1a.slider("Min conf: Bottle",    0.0, 1.0, float(bottle_min),   0.01)
    can_min      = c2a.slider("Min conf: Can",       0.0, 1.0, float(can_min),      0.01)
    foam_min     = c3a.slider("Min conf: Styrofoam", 0.0, 1.0, float(foam_min),     0.01)
    min_area_pct = c4a.slider("Min box area (%)",    0.0, 5.0,  float(min_area_pct), 0.1,
                              help="Ignore tiny boxes by percent of image area.")
    tta = st.toggle("Test time augmentation", value=tta, help="Slower; sometimes reduces false positives.")

# ======================= Load model (optional) =======================
if st.button("Load model"):
    _ = load_model()
    st.success("Model ready.")

# ======================= Detection =======================
def run_detection(image_pil: Image.Image):
    model = load_model()
    bgr = np.array(image_pil.convert("RGB"))[:, :, ::-1]
    results = model.predict(bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False, augment=tta)
    pred = results[0]
    if pred.boxes is None or len(pred.boxes) == 0:
        st.info("No detections")
        return [], {}
    boxes = pred.boxes.xyxy.cpu().numpy()
    scores = pred.boxes.conf.cpu().numpy()
    clsi   = pred.boxes.cls.cpu().numpy().astype(int)

    # Names
    if FORCE_CLASS_NAMES:
        names_map = {i:n for i,n in enumerate(TARGET_NAMES)}
    else:
        nm = getattr(model, "names", None)
        if isinstance(nm, dict): names_map = nm
        elif isinstance(nm, list): names_map = {i:n for i,n in enumerate(nm)}
        else: names_map = {0:"Clear plastic bottle", 1:"Drink can", 2:"Styrofoam piece"}

    per_class_min = {
        "Clear plastic bottle": bottle_min,
        "Drink can":           can_min,
        "Styrofoam piece":     foam_min,
    }

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

    return dets, counts

def draw_and_show(image_pil: Image.Image, dets):
    import cv2
    bgr = np.array(image_pil.convert("RGB"))[:, :, ::-1]
    out = bgr.copy()
    color = (28,160,78)
    H, W = out.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{d["class_name"]} {d["score"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = y1 - 4
        if y_text - th - 4 < 0: y_text = min(y1 + th + 6, H - 2)
        x_text = max(0, min(x1, W - tw - 6))
        cv2.rectangle(out, (x_text, max(0, y_text - th - 4)),
                           (min(x_text + tw + 6, W - 1), min(y_text + 2, H - 1)), color, -1)
        cv2.putText(out, label, (x_text + 3, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    st.image(Image.fromarray(out[:, :, ::-1]), caption="Detections", use_container_width=True)

# Show picked image & (auto) run detection
if image is not None:
    st.image(image, caption="Input", use_container_width=True)
    should_run = auto_run
    if not auto_run:
        should_run = st.button("Run detection")
    if should_run:
        dets, counts = run_detection(image)
        if dets:
            draw_and_show(image, dets)
            detected_labels = sorted({d["class_name"] for d in dets})
            guide_labels = [lbl for lbl in detected_labels if lbl in GUIDE]
            if guide_labels:
                st.subheader(f"Disposal instructions — {city_label}")
                for lbl in guide_labels:
                    show_guidance_card(lbl, counts.get(lbl, 0), GUIDE=GUIDE)
            else:
                st.caption("No local guidance to show for these detections.")
        else:
            st.info("All detections were filtered by thresholds. Try lowering per-class thresholds or min box area.")

# ======================= Impact & SDGs (local images @ 180px) =======================
st.markdown("#### Impact & SDGs")
st.markdown("""
- **Carbon credits (what they are):** A carbon credit represents **1 tonne of CO₂-equivalent** reduced or removed. Credits exist only when a **registered project** follows an **approved methodology** and passes **MRV**; they are then **issued on a registry** (e.g., Gold Standard, Verra, or Japan’s J-Credit).  
- **This app does not issue credits.** It helps people sort properly. Educational CO₂e-avoided estimates are okay, but they’re **not credits**.
""", unsafe_allow_html=True)
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

sdgs = [
    ("sdg12.png", "12 Responsible Consumption & Production"),
    ("sdg11.png", "11 Sustainable Cities & Communities"),
    ("sdg13.png", "13 Climate Action"),
]
st.markdown('<div class="sdg-row">', unsafe_allow_html=True)
for img, caption in sdgs:
    st.markdown('<div class="sdg-card">', unsafe_allow_html=True)
    if os.path.exists(img): st.image(img, width=180)
    else: st.markdown(f"*Missing {img}*")
    st.markdown(f"<div class='txt'>{caption}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
