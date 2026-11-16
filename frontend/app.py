import io
import os

import pymupdf
import pytesseract
import requests
import streamlit as st
from PIL import Image


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PatentMind AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# Tesseract configuration for Windows and Docker/Linux.
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"


# =====================================================
# SESSION STATE
# =====================================================

if "patent_text" not in st.session_state:
    st.session_state.patent_text = ""

if "last_classification" not in st.session_state:
    st.session_state.last_classification = None

if "last_entities" not in st.session_state:
    st.session_state.last_entities = None

if "last_search" not in st.session_state:
    st.session_state.last_search = None

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None


# =====================================================
# HYBRID UI THEME
# Dashboard = light; analysis modules = dark
# =====================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Patent Classification",
        "Entity Extraction",
        "Patent Retrieval",
        "AI Analysis",
    ],
    key="navigation_page",
)

is_dashboard = page == "Dashboard"

if is_dashboard:
    app_background = "#f5f6ff"
    main_text = "#172044"
    muted_text = "#66708f"
    panel_background = "#ffffff"
    panel_border = "#e4e7f3"
    input_background = "#ffffff"
else:
    app_background = "#080f20"
    main_text = "#f2f4ff"
    muted_text = "#a7b1ca"
    panel_background = "#111b31"
    panel_border = "#293754"
    input_background = "#111b31"

st.markdown(
    f"""
    <style>
    :root {{
        --pm-accent: #7655ee;
        --pm-accent-2: #a855f7;
        --pm-bg: {app_background};
        --pm-text: {main_text};
        --pm-muted: {muted_text};
        --pm-panel: {panel_background};
        --pm-border: {panel_border};
    }}

    .stApp {{
        background: var(--pm-bg);
        color: var(--pm-text);
        transition: background 350ms ease, color 350ms ease;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #111a3b 0%, #171b42 58%, #21184d 100%);
        border-right: 1px solid rgba(157, 140, 255, 0.22);
    }}

    [data-testid="stSidebar"] * {{
        color: #f4f2ff;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        border-radius: 10px;
        padding: 7px 9px;
        transition: background 180ms ease, transform 180ms ease;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background: rgba(139, 105, 255, 0.20);
        transform: translateX(3px);
    }}

    .block-container {{
        max-width: 1500px;
        padding: 2rem 2.6rem 3rem;
    }}

    h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {{
        color: var(--pm-text);
    }}

    [data-testid="stCaptionContainer"] {{
        color: var(--pm-muted);
    }}

    .stButton > button {{
        background: linear-gradient(100deg, #6847e8, #9254e9);
        color: #ffffff;
        border: 0;
        border-radius: 10px;
        min-height: 2.65rem;
        padding: 0.55rem 1.2rem;
        font-weight: 650;
        box-shadow: 0 5px 16px rgba(111, 76, 230, 0.20);
        transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
    }}

    .stButton > button:hover {{
        color: #ffffff;
        border: 0;
        transform: translateY(-2px);
        box-shadow: 0 9px 22px rgba(111, 76, 230, 0.32);
        filter: brightness(1.08);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    [data-testid="stMetric"],
    [data-testid="stExpander"],
    [data-testid="stFileUploader"],
    [data-testid="stDataFrame"] {{
        background: var(--pm-panel);
        border: 1px solid var(--pm-border);
        border-radius: 14px;
    }}

    [data-testid="stMetric"] {{
        padding: 16px 18px;
        transition: transform 180ms ease, box-shadow 180ms ease;
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(70, 70, 130, 0.12);
    }}

    [data-testid="stTextArea"] textarea {{
        background: {input_background};
        color: var(--pm-text);
        border: 1px solid var(--pm-border);
        border-radius: 12px;
    }}

    [data-testid="stTextInput"] input {{
        background: {input_background};
        color: var(--pm-text);
        border-radius: 10px;
    }}

    .pm-hero {{
        position: relative;
        overflow: hidden;
        padding: 34px 38px;
        border-radius: 22px;
        color: white;
        background:
            radial-gradient(circle at 82% 20%, rgba(180, 112, 255, .48), transparent 28%),
            linear-gradient(115deg, #111b50 0%, #382477 58%, #6039a8 100%);
        box-shadow: 0 15px 38px rgba(49, 35, 119, .20);
        animation: pm-fade-up 550ms ease both;
    }}

    .pm-hero h1 {{
        color: #ffffff !important;
        font-size: clamp(2rem, 4vw, 3.15rem);
        line-height: 1.1;
        margin: 0 0 12px 0;
        letter-spacing: -1.3px;
    }}

    .pm-hero p {{
        color: #e5e4ff !important;
        font-size: 1.05rem;
        max-width: 720px;
        margin-bottom: 0;
    }}

    .pm-eyebrow {{
        color: #c8baff !important;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }}

    .pm-card {{
        min-height: 175px;
        padding: 22px;
        border-radius: 17px;
        border: 1px solid var(--pm-border);
        background: var(--pm-panel);
        transition: transform 200ms ease, box-shadow 200ms ease;
        animation: pm-fade-up 500ms ease both;
    }}

    .pm-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 13px 28px rgba(78, 67, 150, .13);
    }}

    .pm-card h3 {{
        margin: 8px 0 7px 0;
        font-size: 1.08rem;
        color: var(--pm-text) !important;
    }}

    .pm-card p {{
        color: var(--pm-muted) !important;
        font-size: .92rem;
        margin: 0;
    }}

    .pm-icon {{
        display: inline-flex;
        width: 42px;
        height: 42px;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        font-size: 1.35rem;
        background: rgba(125, 92, 240, .13);
    }}

    .pm-section-title {{
        font-size: 1.25rem;
        font-weight: 750;
        color: var(--pm-text);
        margin: 25px 0 13px;
    }}

    .pm-module-banner {{
        padding: 23px 26px;
        border-radius: 17px;
        border: 1px solid var(--pm-border);
        background: linear-gradient(110deg, var(--pm-panel), rgba(105, 76, 190, .13));
        margin-bottom: 20px;
        animation: pm-fade-up 400ms ease both;
    }}

    .pm-module-banner h1 {{
        margin: 0 0 5px 0;
        font-size: 2rem;
    }}

    .pm-module-banner p {{
        color: var(--pm-muted) !important;
        margin: 0;
    }}

    .pm-note {{
        padding: 14px 17px;
        border-radius: 12px;
        background: rgba(116, 84, 238, .10);
        border: 1px solid rgba(140, 116, 255, .24);
        color: var(--pm-text);
    }}

    @keyframes pm-fade-up {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @media (max-width: 800px) {{
        .block-container {{ padding: 1.2rem 1rem 2rem; }}
        .pm-hero {{ padding: 25px 22px; }}
        .pm-hero h1 {{ font-size: 2rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# SIDEBAR BRANDING
# =====================================================

st.sidebar.markdown(
    """
    <div style="padding:12px 4px 20px 4px;">
      <div style="font-size:2rem;">⚖️ <b>PatentMind AI</b></div>
      <div style="color:#c7c4e8;font-size:.84rem;margin-top:6px;">
        AI-powered Patent Research & Analysis
      </div>
      <hr style="border-color:rgba(190,180,255,.25);margin-top:20px;">
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("PROJECT MODULES")
st.sidebar.markdown(
    """
    <div style="font-size:.82rem;color:#c7c4e8;line-height:1.8;padding:8px 2px;">
      <b>ML</b> · Classification<br>
      <b>NLP</b> · Entity Extraction<br>
      <b>IR</b> · SBERT + FAISS<br>
      <b>RAG</b> · Groq LLM
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()
st.sidebar.caption("PatentMind-AI · Research prototype")


# =====================================================
# API HELPER
# =====================================================

def call_api(endpoint, text):
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            json={"text": text},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"API returned an error ({response.status_code}): {detail}")
        return None

    except requests.exceptions.RequestException as exc:
        st.error(f"API connection error: {exc}")
        return None


# =====================================================
# PDF TEXT EXTRACTION AND OCR
# =====================================================

def extract_pdf_text(uploaded_file):
    extracted_pages = []

    try:
        document = pymupdf.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf",
        )

        total_pages = len(document)
        if total_pages == 0:
            st.warning("The uploaded PDF contains no pages.")
            document.close()
            return ""

        progress = st.progress(0)
        status = st.empty()

        for page_number, page in enumerate(document):
            status.write(
                f"Processing page {page_number + 1} of {total_pages}..."
            )

            page_text = page.get_text("text")

            # OCR is used only when normal PDF text extraction finds no text.
            if not page_text.strip():
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                page_text = pytesseract.image_to_string(image)

            extracted_pages.append(
                f"--- Page {page_number + 1} ---\n{page_text}"
            )
            progress.progress((page_number + 1) / total_pages)

        document.close()
        status.success("PDF processing completed.")
        return "\n\n".join(extracted_pages).strip()

    except Exception as exc:
        st.error(f"PDF extraction failed: {exc}")
        return ""


# =====================================================
# COMMON PATENT INPUT
# =====================================================

def get_patent_text():
    st.markdown(
        '<div class="pm-note">Upload a patent PDF to extract its text, '
        'or paste patent text directly below.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Patent PDF",
        type=["pdf"],
        key=f"pdf_{page}",
    )

    if uploaded_file:
        st.caption(
            f"Selected file: {uploaded_file.name} · "
            f"{uploaded_file.size / (1024 * 1024):.2f} MB"
        )
        if st.button("Extract PDF Text", key=f"extract_{page}"):
            with st.spinner("Extracting patent text and checking scanned pages..."):
                extracted = extract_pdf_text(uploaded_file)

            if extracted:
                st.session_state.patent_text = extracted
                st.success(
                    f"Extracted {len(extracted):,} characters successfully."
                )

    manual_text = st.text_area(
        "Or enter patent text manually",
        height=190,
        key=f"manual_{page}",
        placeholder="Paste the patent abstract, claims, or description here...",
    )

    if manual_text.strip():
        return manual_text.strip()

    return st.session_state.patent_text.strip()


# =====================================================
# PAGE HEADER HELPER
# =====================================================

def module_header(title, subtitle, icon):
    st.markdown(
        f"""
        <div class="pm-module-banner">
          <div style="font-size:1.7rem;margin-bottom:5px;">{icon}</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# DASHBOARD — LIGHT THEME
# =====================================================

if page == "Dashboard":
    st.markdown(
        """
        <div class="pm-hero">
          <div class="pm-eyebrow">AI-powered patent intelligence</div>
          <h1>Welcome to PatentMind-AI</h1>
          <p>
            Classify patents, extract technical entities, discover similar
            inventions, and generate evidence-aware research insights —
            all in one workspace.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="pm-section-title">Your research workspace</div>',
                unsafe_allow_html=True)

    cards = [
        ("🧩", "Patent Classification",
         "Predict the CPC section (A–Y) using the trained TF-IDF and Logistic Regression pipeline."),
        ("🔎", "Entity Extraction",
         "Identify technical entities such as components, materials, organisations, and locations."),
        ("📚", "Patent Retrieval",
         "Find semantically similar patent abstracts using Sentence-BERT and FAISS."),
        ("✨", "AI Analysis (RAG)",
         "Generate a structured research summary using retrieved patent context and Groq LLM."),
    ]

    cols = st.columns(4)
    for col, (icon, title, description) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="pm-card">
                  <div class="pm-icon">{icon}</div>
                  <h3>{title}</h3>
                  <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="pm-section-title">Project at a glance</div>',
                unsafe_allow_html=True)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Dataset records", "9,000")
    metric_cols[1].metric("CPC sections", "9")
    metric_cols[2].metric("Core modules", "4")
    metric_cols[3].metric("Deployment", "Docker")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### End-to-end workflow")
        st.markdown(
            """
            <div class="pm-card" style="min-height:0;">
              <p style="font-size:1rem;line-height:2;">
                <b>01</b> &nbsp; Patent PDF / text input<br>
                <b>02</b> &nbsp; Text extraction and OCR<br>
                <b>03</b> &nbsp; Classification and entity extraction<br>
                <b>04</b> &nbsp; SBERT embeddings and FAISS retrieval<br>
                <b>05</b> &nbsp; RAG-based patent research analysis
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### Technology stack")
        st.markdown(
            """
            <div class="pm-card" style="min-height:0;">
              <p style="font-size:.95rem;line-height:2;">
                <b>ML:</b> Scikit-learn<br>
                <b>NLP:</b> spaCy, Sentence Transformers<br>
                <b>Vector search:</b> FAISS<br>
                <b>LLM:</b> Groq API<br>
                <b>Backend:</b> FastAPI<br>
                <b>Frontend:</b> Streamlit<br>
                <b>MLOps:</b> DVC, MLflow, Docker
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "PatentMind-AI is a research-support prototype. Its outputs should "
        "be reviewed by qualified patent professionals; it does not provide "
        "legal advice or determine patentability."
    )


# =====================================================
# PATENT CLASSIFICATION
# =====================================================

elif page == "Patent Classification":
    module_header(
        "Patent Classification",
        "Predict the Cooperative Patent Classification (CPC) section for a patent.",
        "🧩",
    )

    text = get_patent_text()

    if st.button("Classify Patent", type="primary"):
        if not text:
            st.warning("Please upload a PDF or enter patent text.")
        else:
            with st.spinner("Running the classification model..."):
                result = call_api("classify", text)

            if result:
                st.session_state.last_classification = result
                st.success("Classification completed.")

result = st.session_state.last_classification

if result:
    st.markdown("### Prediction")

    # Main prediction
    category = result.get("predicted_category", "N/A")
    category_name = result.get("category_name", "Unknown Category")
    description = result.get("description", "No description available.")

    st.markdown(
        f"""
        <div class="result-card">
            <p>Predicted CPC Category</p>
            <h1>{category}</h1>
            <h3>{category_name}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("Classification completed successfully!")

    # Top 3 predictions
    top_predictions = result.get("top_predictions", [])

    if top_predictions:
        st.markdown("### 📊 Top 3 Possible Categories")
        st.caption(
            "Probability scores generated by the trained Logistic Regression model."
        )

        for item in top_predictions:
            code = item.get("category", "N/A")
            name = item.get("name", "Unknown")
            probability = float(item.get("probability", 0))

            st.markdown(
                f"**CPC {code} — {name}**"
            )

            st.progress(
                max(0.0, min(probability / 100, 1.0))
            )

            st.caption(f"Model probability: {probability:.2f}%")

    else:
        st.info(
            "Alternative category probabilities are unavailable "
            "for this classifier."
        )

    st.caption(
        "Prediction generated using the trained TF-IDF + "
        "Logistic Regression pipeline. CPC descriptions provide "
        "general guidance, not a legal determination."
    )

# =====================================================
# ENTITY EXTRACTION
# =====================================================

elif page == "Entity Extraction":
    module_header(
        "Patent Entity Extraction",
        "Extract technical and named entities from patent text.",
        "🔎",
    )

    text = get_patent_text()

    if st.button("Extract Entities", type="primary"):
        if not text:
            st.warning("Please provide patent text.")
        else:
            with st.spinner("Extracting entities..."):
                result = call_api("extract-entities", text)

            if result:
                st.session_state.last_entities = result
                st.success("Entity extraction completed.")


result = st.session_state.last_entities

if result:
    entities = result.get("entities", [])

    st.markdown("### Extracted Patent Entities")

    # Handle dictionary or list response
    if isinstance(entities, dict):
        entity_groups = entities

    elif isinstance(entities, list):
        entity_groups = {}

        for entity in entities:
            if isinstance(entity, dict):
                label = (
                    entity.get("label")
                    or entity.get("entity_group")
                    or entity.get("type")
                    or "OTHER"
                )

                value = (
                    entity.get("text")
                    or entity.get("value")
                    or entity.get("word")
                )

                if value:
                    entity_groups.setdefault(label, []).append(value)

    else:
        entity_groups = {}

    if entity_groups:
        total_entities = sum(
            len(values) if isinstance(values, list) else 1
            for values in entity_groups.values()
        )

        st.metric("Total Extracted Entities", total_entities)

        for entity_type, values in entity_groups.items():

            if not isinstance(values, list):
                values = [values]

            # Remove duplicate values
            values = list(dict.fromkeys(
                str(value) for value in values if value
            ))

            with st.expander(
                f"{entity_type.replace('_', ' ').title()} ({len(values)})",
                expanded=True
            ):
                if values:
                    st.write(" • ".join(values))
                else:
                    st.caption("No entities found.")

    else:
        st.info("No entities were detected in the provided patent text.")

# =====================================================
# PATENT RETRIEVAL
# =====================================================

elif page == "Patent Retrieval":
    module_header(
        "Similar Patent Search",
        "Search the BIGPATENT index for semantically similar patent abstracts.",
        "📚",
    )

    text = get_patent_text()

    if st.button("Search Similar Patents", type="primary"):
        if not text:
            st.warning("Please provide patent text.")
        else:
            with st.spinner("Generating embeddings and searching FAISS..."):
                result = call_api("search", text)

            if result:
                st.session_state.last_search = result
                st.success("Similar patents retrieved.")

    result = st.session_state.last_search
    if result:
        patents = result.get("similar_patents", [])
        st.markdown(f"### Top {len(patents)} similar patents")

        for index, patent in enumerate(patents, start=1):
            category = patent.get("category", "N/A")
            score = patent.get("similarity_score", "N/A")
            with st.expander(
                f"Patent {index}  ·  CPC {category}  ·  Similarity {score}"
            ):
                st.write("**CPC Category:**", category)
                st.write("**Similarity Score:**", score)
                st.write("**Abstract:**")
                st.write(patent.get("abstract", "N/A"))


# =====================================================
# AI ANALYSIS / RAG
# =====================================================

elif page == "AI Analysis":
    module_header(
        "AI Patent Analysis",
        "Retrieval-Augmented Generation using similar patents and the Groq LLM.",
        "✨",
    )

    text = get_patent_text()

    st.info(
        "For best results, provide a focused patent abstract or relevant "
        "description. Very long full-document inputs may exceed the LLM "
        "provider's request limits."
    )

    if st.button("Analyze Patent", type="primary"):
        if not text:
            st.warning("Please provide patent text.")
        else:
            with st.spinner("Retrieving similar patents and generating analysis..."):
                result = call_api("analyze", text)

            if result:
                st.session_state.last_analysis = result
                st.success("AI analysis completed.")

    result = st.session_state.last_analysis
    if result:
        analysis = result.get("analysis", "")
        st.markdown("### Research analysis")
        st.markdown(analysis)

        st.caption(
            "AI-generated research support only. Verify all statements against "
            "the cited patent documents and seek qualified legal review."
        )
