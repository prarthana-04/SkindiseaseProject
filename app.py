# ================= IMPORTS =================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
from skimage.io import imread
from skimage import color
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.util import img_as_ubyte
from scipy.stats import skew
import base64

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(
            135deg,
            #f5f3ff 0%,   /* lavender */
            #ede9fe 35%,  /* soft lilac */
            #eef6ff 70%,  /* pastel blue */
            #f8fbff 100%  /* airy light blue */
        ) !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent !important;
    }

    .block-container {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SkinCheck AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= PATHS =================
BASE = Path.cwd()
MODEL_DIR = BASE / "models"
FEATURES_CSV = BASE / "features" / "features.csv"
TEMP_IMG = BASE / "temp.jpg"

# ================= LABEL MAP =================
LABEL_MAP = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}

# ================= GLOBAL CSS =================
st.markdown("""
<style>

/* ---------- BACKGROUND ---------- */
.stApp {
    background: linear-gradient(
        135deg,
        #fff1f5 0%,
        #f8fbff 45%,
        #eef6ff 100%
    ) !important;
    color: #111827 !important;
}

/* ---------- TITLES ---------- */
.title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    color: #1e3a8a !important;
}
.subtitle {
    font-size: 22px;
    text-align: center;
    color: #334155 !important;
    margin-bottom: 30px;
}

/* ---------- SECTION HEADERS ---------- */
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 14px;
}

/* ---------- RESULT TEXT ---------- */
.result-text {
    font-size: 24px;
    margin-bottom: 12px;
}
.highlight {
    color: #047857 !important;
    font-weight: 800;
}

/* ---------- ANALYZE BUTTON ---------- */
.stButton > button {
    background-color: #1e40af !important;
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    padding: 10px 26px !important;
    border-radius: 12px !important;
}
.stButton > button span {
    color: #ffffff !important;
}

/* ---------- FILE UPLOADER ---------- */
section[data-testid="stFileUploader"] * {
    color: #111827 !important;
    font-size: 18px !important;
}
div[data-testid="stFileUploaderDropzone"] {
    background-color: #e5e7eb !important;
    border: 2px dashed #94a3b8 !important;
}

/* ---------- EXPANDER (FIXED – NO STRETCHING) ---------- */
div[data-testid="stExpander"] {
    border: none !important;
}

div[data-testid="stExpander"] summary {
    background-color: #1f2937 !important;
    padding: 8px 14px !important;
    border-radius: 8px !important;
}

div[data-testid="stExpander"] summary span {
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

div[data-testid="stExpander"] > div {
    padding: 10px 14px !important;
    background-color: #ffffff !important;
}

/* model rows */
.model-row {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 8px;
    color: #111827;
}

/* ---------- DISCLAIMER ---------- */
div[data-testid="stAlert"] {
    background-color: #fef3c7 !important;
    border-radius: 10px !important;
}
div[data-testid="stAlert"] * {
    color: #92400e !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# ================= FEATURE EXTRACTION =================
def extract_features(img):
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    gray = color.rgb2gray(img)

    cm = []
    for c in range(3):
        vals = img[..., c].ravel()
        cm.extend([np.mean(vals), np.std(vals), skew(vals)])

    glcm = graycomatrix(img_as_ubyte(gray), [1], [0], symmetric=True, normed=True)
    glcm_feats = [graycoprops(glcm, p)[0, 0]
                  for p in ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity']]

    lbp = local_binary_pattern(img_as_ubyte(gray), 8, 1, method='uniform')
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    return np.concatenate([cm, glcm_feats, hist]).reshape(1, -1)

# ================= LOAD MODELS =================
def load_models():
    models = {}
    for f in MODEL_DIR.glob("*.joblib"):
        name = f.stem.lower()
        if "svm" in name:
            models["SVM"] = joblib.load(f)
        elif "rf" in name or "random" in name:
            models["Random Forest"] = joblib.load(f)
        elif "knn" in name:
            models["KNN"] = joblib.load(f)
    return models

def get_classes():
    if FEATURES_CSV.exists():
        return list(pd.read_csv(FEATURES_CSV)["dx"].unique())
    return list(LABEL_MAP.keys())

# ================= HEADER =================
st.markdown('<div class="title">🔬 SkinCheck AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Simple and reliable AI-based skin screening</div>', unsafe_allow_html=True)

# ================= INPUT =================
st.markdown('<div class="section-title">📥 Upload or Capture Image</div>', unsafe_allow_html=True)

cam_col, upload_col = st.columns([1, 1.4])

with cam_col:
    st.markdown(
        "<div style='text-align:center; font-size:24px; font-weight:700;'>📷 Camera</div>",
        unsafe_allow_html=True
    )
    camera_img = st.camera_input("camera", label_visibility="collapsed")

with upload_col:
    st.markdown(
        "<div style='text-align:center; font-size:24px; font-weight:700;'>📁 Upload Image</div>",
        unsafe_allow_html=True
    )
    uploaded_img = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

analyze = st.button("🔍 Analyze Image")

# ================= ANALYSIS =================
if analyze:
    if camera_img is None and uploaded_img is None:
        st.warning("Please upload or capture an image.")
    else:
        image_bytes = camera_img.getvalue() if camera_img else uploaded_img.getvalue()

        with open(TEMP_IMG, "wb") as f:
            f.write(image_bytes)

        img = imread(TEMP_IMG)
        features = extract_features(img)

        models = load_models()
        classes = get_classes()
        results = {}

        for name, model in models.items():
            pred = model.predict(features)[0]
            conf = model.predict_proba(features).max() if hasattr(model, "predict_proba") else None
            label = classes[int(pred)] if isinstance(pred, (int, np.integer)) else pred
            results[name] = {"label": label, "confidence": conf}

        final_code = Counter([v["label"] for v in results.values()]).most_common(1)[0][0]
        final_label = LABEL_MAP.get(final_code, final_code)

        best_model = max(
            results.items(),
            key=lambda x: x[1]["confidence"] if x[1]["confidence"] is not None else 0
        )

        # ================= RESULT =================
        st.markdown('<div class="section-title">📊 Analysis Result</div>', unsafe_allow_html=True)

        img_col, text_col = st.columns([1, 2])

        with img_col:
            encoded = base64.b64encode(open(TEMP_IMG, "rb").read()).decode()
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <img src="data:image/jpeg;base64,{encoded}"
                         style="width:230px; border-radius:8px;" />
                    <div style="margin-top:6px; font-size:18px; font-weight:700;">
                        Analyzed Image
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with text_col:
            st.markdown(
                f"<div class='result-text'>Final Prediction: "
                f"<span class='highlight'>{final_label}</span></div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='result-text'>Best Model: "
                f"<span class='highlight'>{best_model[0]}</span></div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='result-text'>Confidence: "
                f"<span class='highlight'>{best_model[1]['confidence']*100:.2f}%</span></div>",
                unsafe_allow_html=True
            )

        with st.expander("🔽 Model-wise Confidence"):
            for m, v in results.items():
                if v["confidence"] is not None:
                    st.markdown(
                        f"<div class='model-row'>"
                        f"{m}: <b>{LABEL_MAP.get(v['label'], v['label'])}</b> "
                        f"({v['confidence']*100:.2f}%)"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='model-row'>"
                        f"{m}: <b>{LABEL_MAP.get(v['label'], v['label'])}</b> (N/A)"
                        f"</div>",
                        unsafe_allow_html=True
                    )

# ================= DISCLAIMER =================
st.warning(
    "⚠️ This application provides AI-based screening suggestions only. "
    "It is not a medical diagnosis. Always consult a certified dermatologist."
)
