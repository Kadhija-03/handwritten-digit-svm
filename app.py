
import streamlit as st
import requests
from PIL import Image
import io
import hashlib


# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="DigitAI | Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)


import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

# =====================================
# 2. DARK THEME CSS
# =====================================

st.html("""
<style>

.stApp {
    background-color: #111111;
    color: #FFFFFF;
}

[data-testid="stAppViewContainer"] {
    background-color: #111111;
}

[data-testid="stHeader"] {
    background-color: #111111;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Header */

.brand {
    color: #FFFFFF;
    font-size: 21px;
    font-weight: 750;
    letter-spacing: 1px;
}

.brand span {
    color: #60A5FA;
}

/* Main heading */

.main-title {
    color: #FFFFFF;
    font-size: 34px;
    font-weight: 700;
    margin-top: 45px;
    margin-bottom: 10px;
}

.main-description {
    color: #A3A3A3;
    font-size: 15px;
    line-height: 1.7;
}

/* Model information */

.model-info {
    display: flex;
    gap: 55px;
    flex-wrap: wrap;
    margin-top: 28px;
    margin-bottom: 32px;
    padding: 18px 0;
    border-top: 1px solid #333333;
    border-bottom: 1px solid #333333;
}

.info-label {
    color: #999999;
    font-size: 11px;
    margin-bottom: 6px;
    letter-spacing: 1px;
}

.info-value {
    color: #FFFFFF;
    font-size: 15px;
    font-weight: 600;
}

/* Section headings */

.section-title {
    color: #FFFFFF;
    font-size: 18px;
    font-weight: 650;
    margin-bottom: 7px;
}

.section-description {
    color: #999999;
    font-size: 13px;
    margin-bottom: 18px;
}

/* Upload area */

[data-testid="stFileUploader"] {
    background: #1D1D1D;
    border: 1px dashed #444444;
    border-radius: 8px;
    padding: 12px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
    color: #FFFFFF;
}

[data-testid="stFileUploader"] button {
    background: #292929;
    color: #60A5FA;
    border: 1px solid #444444;
}

/* Image */

[data-testid="stImage"] {
    border-radius: 8px;
    overflow: hidden;
}

/* Prediction button */

.stButton > button {
    width: 100%;
    background: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 7px;
    min-height: 43px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #1D4ED8;
    color: #FFFFFF;
    border: none;
}

/* Result panel */

.result-box {
    background: #202020;
    border: 1px solid #383838;
    border-radius: 9px;
    min-height: 230px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 20px;
}

.result-label {
    color: #A3A3A3;
    font-size: 11px;
    letter-spacing: 1.5px;
}

.result-digit {
    color: #60A5FA;
    font-size: 100px;
    font-weight: 750;
    line-height: 1.3;
}

.result-caption {
    color: #A3A3A3;
    font-size: 13px;
}

/* Empty result */

.empty-result {
    background: #1A1A1A;
    border: 1px solid #333333;
    border-radius: 9px;
    min-height: 230px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 20px;
}

.empty-result p {
    color: #999999;
    font-size: 13px;
}

/* Footer */

.footer {
    border-top: 1px solid #333333;
    margin-top: 50px;
    padding-top: 18px;
    text-align: center;
    color: #777777;
    font-size: 12px;
}

</style>
""")


# =====================================
# 3. SESSION STATE
# =====================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# =====================================
# 4. HEADER
# =====================================

st.html("""
<div class="brand">
    DIGIT<span>AI</span>
</div>

<div class="main-title">
    Handwritten Digit Recognition
</div>

<div class="main-description">
    Upload a handwritten digit image and let our
    machine learning model identify the digit.
</div>

<div class="model-info">

    <div>
        <div class="info-label">MODEL</div>
        <div class="info-value">RBF Support Vector Machine</div>
    </div>

    <div>
        <div class="info-label">TEST ACCURACY</div>
        <div class="info-value">98.26%</div>
    </div>

    <div>
        <div class="info-label">SUPPORTED DIGITS</div>
        <div class="info-value">0 – 9</div>
    </div>

</div>
""")


# =====================================
# 5. UPLOAD AND RESULT LAYOUT
# =====================================

left, right = st.columns(2, gap="large")


# =====================================
# 6. IMAGE UPLOAD
# =====================================

with left:

    st.html("""
    <div class="section-title">Upload image</div>

    <div class="section-description">
        Select an image containing a single handwritten digit.
    </div>
    """)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        original_bytes = uploaded_file.getvalue()

        current_hash = hashlib.md5(
            original_bytes
        ).hexdigest()

        if current_hash != st.session_state.file_hash:
            st.session_state.prediction = None
            st.session_state.file_hash = current_hash

        image = Image.open(
            io.BytesIO(original_bytes)
        ).convert("RGB")

        st.write("")

        st.image(
            image,
            caption="Uploaded image",
            width=250
        )

        st.write("")

        predict_button = st.button(
            "Recognize Digit",
            type="primary"
        )

    else:

        st.html("""
        <div class="empty-result">
            <div style="font-size:30px;color:#60A5FA;">
                ↑
            </div>
            <p>
                Your uploaded image preview will appear here.
            </p>
        </div>
        """)

        predict_button = False


# =====================================
# 7. FASTAPI PREDICTION
# =====================================

if uploaded_file is not None and predict_button:

    with st.spinner("Recognizing digit..."):

        try:

            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes.seek(0)

            files = {
                "file": (
                    uploaded_file.name,
                    image_bytes,
                    "image/png"
                )
            }

            response = requests.post(
                API_URL,
                files=files,
                timeout=30
            )

            response.raise_for_status()

            result = response.json()

            st.session_state.prediction = int(
                result["predicted_digit"]
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI. "
                "Please start the API server."
            )

        except requests.exceptions.Timeout:

            st.error("Prediction request timed out.")

        except requests.exceptions.HTTPError:

            try:
                message = response.json().get(
                    "detail", "Unknown API error"
                )
            except Exception:
                message = response.text

            st.error(f"Prediction failed: {message}")

        except Exception as e:

            st.error(f"Unexpected error: {e}")


# =====================================
# 8. PREDICTION RESULT
# =====================================

with right:

    st.html("""
    <div class="section-title">Prediction result</div>

    <div class="section-description">
        The recognized digit will appear below.
    </div>
    """)

    if st.session_state.prediction is not None:

        digit = st.session_state.prediction

        st.html(f"""
        <div class="result-box">

            <div class="result-label">
                PREDICTED DIGIT
            </div>

            <div class="result-digit">
                {digit}
            </div>

            <div class="result-caption">
                Recognition successful
            </div>

        </div>
        """)

        st.html(f"""
        <div style="
            background:#172B45;
            border-radius:7px;
            padding:12px;
            margin-top:12px;
            color:#93C5FD;
            font-size:14px;
        ">
            ✓ &nbsp; Predicted digit: {digit}
        </div>
        """)

    else:

        st.html("""
        <div class="empty-result">

            <div style="font-size:30px;color:#60A5FA;">
                ◎
            </div>

            <p>
                Your prediction will appear here.
            </p>

        </div>
        """)


# =====================================
# 9. FOOTER
# =====================================

st.html("""
<div class="footer">
    DIGITAI &nbsp; · &nbsp;
    Handwritten Digit Recognition System
    <br><br>
    Python · OpenCV · Scikit-learn · FastAPI · Streamlit
</div>
""")
