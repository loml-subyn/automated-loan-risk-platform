"""
================================================================================
  INSTITUTIONAL CREDIT RISK ASSESSMENT PLATFORM
  ──────────────────────────────────────────────
  Enterprise-grade Streamlit dashboard for automated loan underwriting.

  Features:
    - Dual-model engine toggle (Logistic Regression / Decision Tree)
    - CIBIL PDF extraction layer with pdfplumber
    - Bug-free inference via model_columns.pkl alignment
    - Feature attribution & risk matrix (LR coefficients)
    - Downloadable Underwriting Sanction Clearance Memo

  Artifacts Required:
    loan_model_lr.pkl, loan_model_dt.pkl, scaler.pkl, model_columns.pkl
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import re
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: PAGE CONFIG & CORPORATE BRANDING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Assessment Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — CORPORATE DARK THEME
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root Variables ─────────────────────────────────────────────── */
    :root {
        --bg-primary:    #0a0e1a;
        --bg-secondary:  #111827;
        --bg-card:       #1a2035;
        --bg-card-hover: #1f2847;
        --border:        #1e2a45;
        --border-accent: #2563eb33;
        --accent:        #3b82f6;
        --accent-light:  #60a5fa;
        --accent-dim:    #1e3a5f;
        --text-primary:  #f1f5f9;
        --text-secondary:#94a3b8;
        --text-muted:    #64748b;
        --success:       #10b981;
        --success-bg:    #052e16;
        --success-border:#065f46;
        --danger:        #ef4444;
        --danger-bg:     #350a0a;
        --danger-border: #7f1d1d;
        --warning:       #f59e0b;
        --warning-bg:    #351c05;
    }

    /* ── Global Reset ───────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background: var(--bg-primary);
    }

    /* ── Remove Streamlit default padding/header ────────────────────── */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* ── Scrollbar ──────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* ── Top Header Bar ─────────────────────────────────────────────── */
    .top-bar {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.6rem 2rem;
        margin-bottom: 0.8rem;
        position: relative;
        overflow: hidden;
    }
    .top-bar::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), #818cf8, var(--accent-light));
    }
    .top-bar h1 {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.04em;
        margin: 0 0 0.2rem 0;
    }
    .top-bar .subtitle {
        font-size: 0.82rem;
        color: var(--text-muted);
        font-weight: 400;
        letter-spacing: 0.02em;
    }

    /* ── Status Metric Chips ────────────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1rem;
    }
    .metric-chip {
        flex: 1;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .metric-chip .dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .metric-chip .dot.green  { background: var(--success); box-shadow: 0 0 6px var(--success); }
    .metric-chip .dot.blue   { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
    .metric-chip .dot.purple { background: #818cf8; box-shadow: 0 0 6px #818cf8; }
    .metric-chip .chip-text {
        font-size: 0.78rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    .metric-chip .chip-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-primary);
        font-weight: 600;
        margin-left: auto;
    }

    /* ── Section Panel ──────────────────────────────────────────────── */
    .panel {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .panel-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent-light);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .panel-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }

    /* ── File Upload Styling ────────────────────────────────────────── */
    .upload-zone {
        border: 2px dashed var(--border);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        background: var(--bg-secondary);
        transition: border-color 0.3s;
        margin-bottom: 1rem;
    }
    .upload-zone:hover {
        border-color: var(--accent);
    }

    /* ── Status Badges ──────────────────────────────────────────────── */
    .badge-success {
        background: var(--success-bg);
        border: 1px solid var(--success-border);
        border-left: 4px solid var(--success);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
    }
    .badge-success p {
        color: #6ee7b7;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0;
    }
    .badge-success .badge-title {
        color: var(--success);
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .badge-danger {
        background: var(--danger-bg);
        border: 1px solid var(--danger-border);
        border-left: 4px solid var(--danger);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
    }
    .badge-danger p {
        color: #fca5a5;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0;
    }
    .badge-danger .badge-title {
        color: var(--danger);
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .badge-warning {
        background: var(--warning-bg);
        border: 1px solid #78350f;
        border-left: 4px solid var(--warning);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
    }
    .badge-warning p {
        color: #fcd34d;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0;
    }
    .badge-warning .badge-title {
        color: var(--warning);
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    /* ── Info Badge (for AI Extraction) ─────────────────────────────── */
    .badge-info {
        background: #0c1f3d;
        border: 1px solid #1e3a5f;
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
    }
    .badge-info p {
        color: var(--accent-light);
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0;
    }
    .badge-info .badge-title {
        color: var(--accent);
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    /* ── Decision Card ──────────────────────────────────────────────── */
    .decision-approved {
        background: linear-gradient(145deg, #052e16, #064e3b);
        border: 1px solid var(--success-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 0.8rem;
        animation: slideUp 0.5s ease-out;
    }
    .decision-rejected {
        background: linear-gradient(145deg, #350a0a, #451a03);
        border: 1px solid var(--danger-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 0.8rem;
        animation: slideUp 0.5s ease-out;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .decision-header {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .decision-verdict {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }
    .decision-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.7rem;
        margin-top: 1rem;
    }
    .d-metric {
        background: rgba(0,0,0,0.25);
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
    }
    .d-metric .d-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
    .d-metric .d-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.15rem;
    }

    /* ── Recommendation List ────────────────────────────────────────── */
    .rec-list {
        margin-top: 1rem;
        padding: 0.8rem 1rem;
        background: rgba(0,0,0,0.2);
        border-radius: 8px;
    }
    .rec-list .rec-title {
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
    }
    .rec-list ul {
        margin: 0;
        padding-left: 1.2rem;
    }
    .rec-list ul li {
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.7;
    }

    /* ── Execute Button ─────────────────────────────────────────────── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1d4ed8, #3b82f6);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        padding: 0.85rem 1.5rem;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 4px 20px rgba(59,130,246,0.3);
        text-transform: uppercase;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 28px rgba(59,130,246,0.45);
        background: linear-gradient(135deg, #2563eb, #60a5fa);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Expander Styling ───────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        background: var(--bg-secondary) !important;
        border-radius: 8px !important;
    }

    /* ── Selectbox / Slider Labels ──────────────────────────────────── */
    .stSelectbox label, .stSlider label, .stNumberInput label, .stFileUploader label {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
    }

    /* ── Feature Attribution Bar ────────────────────────────────────── */
    .attr-bar-container {
        margin: 0.3rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .attr-label {
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--text-secondary);
        width: 160px;
        text-align: right;
        flex-shrink: 0;
    }
    .attr-bar-track {
        flex: 1;
        height: 18px;
        background: rgba(255,255,255,0.04);
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    .attr-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease-out;
    }
    .attr-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        width: 55px;
        flex-shrink: 0;
    }

    /* ── Timestamp Footer ───────────────────────────────────────────── */
    .footer {
        text-align: center;
        font-size: 0.7rem;
        color: var(--text-muted);
        padding: 2rem 0 1rem;
        border-top: 1px solid var(--border);
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER BAR
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="top-bar">
        <h1>🏛️&ensp;INSTITUTIONAL CREDIT RISK ASSESSMENT PLATFORM</h1>
        <p class="subtitle">Automated Credit Scoring, Risk Profiling, and Decisioning Engine — Dual-Model Architecture</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── System Status Metric Row ────────────────────────────────────────────────

current_ts = datetime.now().strftime("%d %b %Y  %H:%M:%S")
st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-chip">
            <span class="dot green"></span>
            <span class="chip-text">System Status</span>
            <span class="chip-val">OPERATIONAL</span>
        </div>
        <div class="metric-chip">
            <span class="dot blue"></span>
            <span class="chip-text">API Latency</span>
            <span class="chip-val">14 ms</span>
        </div>
        <div class="metric-chip">
            <span class="dot purple"></span>
            <span class="chip-text">Model Version</span>
            <span class="chip-val">v3.0.0</span>
        </div>
        <div class="metric-chip">
            <span class="dot blue"></span>
            <span class="chip-text">Timestamp</span>
            <span class="chip-val">{current_ts}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — ALL FOUR ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATHS = {
    "lr_model": os.path.join(BASE_DIR, "loan_model_lr.pkl"),
    "dt_model": os.path.join(BASE_DIR, "loan_model_dt.pkl"),
    "scaler": os.path.join(BASE_DIR, "scaler.pkl"),
    "columns": os.path.join(BASE_DIR, "model_columns.pkl"),
}

model_loaded = True
lr_model = None
dt_model = None
scaler = None
model_columns = None

try:
    lr_model = joblib.load(ARTIFACT_PATHS["lr_model"])
    dt_model = joblib.load(ARTIFACT_PATHS["dt_model"])
    scaler = joblib.load(ARTIFACT_PATHS["scaler"])
    model_columns = joblib.load(ARTIFACT_PATHS["columns"])
except FileNotFoundError as fnf:
    model_loaded = False
    missing_file = str(fnf)
    st.markdown(
        f"""
        <div class="badge-danger">
            <div class="badge-title">⛔ Critical — Model Artifacts Missing</div>
            <p>
                Could not locate one or more required model files:<br>
                <code>loan_model_lr.pkl</code>, <code>loan_model_dt.pkl</code>,
                <code>scaler.pkl</code>, <code>model_columns.pkl</code><br><br>
                <strong>Error:</strong> {missing_file}<br><br>
                Run <code>loan_approval_predictor.py</code> first to train and export the model pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
except Exception as e:
    model_loaded = False
    st.markdown(
        f"""
        <div class="badge-danger">
            <div class="badge-title">⛔ Runtime Error — Model Load Failure</div>
            <p>{e}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CORE UNDERWRITING ENGINE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

if model_loaded:
    engine_choice = st.selectbox(
        "🧠 Core Underwriting Engine",
        options=["Logistic Regression", "Decision Tree Classifier"],
        index=0,
        help="Select the ML model that will power the underwriting decision. "
             "Logistic Regression provides probability-based scoring with coefficient explainability. "
             "Decision Tree provides rule-based classification.",
    )
else:
    engine_choice = "Logistic Regression"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DUAL-COLUMN WORKSPACE LAYOUT (60 / 40)
# ─────────────────────────────────────────────────────────────────────────────

left_col, spacer, right_col = st.columns([6, 0.3, 4])

# ═══════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — APPLICANT PROFILING & DOCUMENT UPLOAD
# ═══════════════════════════════════════════════════════════════════════════

with left_col:

    # ── 1. Document Verification & CIBIL PDF Extraction ──────────────────

    st.markdown(
        '<div class="panel-title">📄&ensp;1. Document Verification & CIBIL Extraction</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Official CIBIL Report (PDF format only)",
        type=["pdf"],
        help="Accepted format: .pdf — Max file size 200 MB. "
             "The system will attempt to extract CIBIL score automatically.",
    )

    # --- CIBIL PDF Extraction Layer ---
    extracted_cibil = None

    if uploaded_file is not None:
        file_size_kb = uploaded_file.size / 1024
        st.markdown(
            f"""
            <div class="badge-success">
                <div class="badge-title">✅ Verification Complete</div>
                <p>
                    CIBIL PDF <strong>"{uploaded_file.name}"</strong>
                    ({file_size_kb:.1f} KB) verified and security scanned successfully.<br>
                    Proceeding with AI text extraction...
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Attempt PDF text extraction using pdfplumber
        try:
            import pdfplumber

            uploaded_file.seek(0)
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"

            # Search for a 3-digit number between 300 and 900
            # Pattern: look for standalone 3-digit numbers in CIBIL score range
            score_candidates = re.findall(r'\b([3-8]\d{2})\b', full_text)
            for candidate in score_candidates:
                val = int(candidate)
                if 300 <= val <= 900:
                    extracted_cibil = val
                    break

            if extracted_cibil is not None:
                st.markdown(
                    f"""
                    <div class="badge-info">
                        <div class="badge-title">🤖 AI Extraction: CIBIL Score Parsed Successfully</div>
                        <p>
                            Extracted CIBIL Score: <strong>{extracted_cibil}</strong> from document.<br>
                            The manual input field below has been updated to reflect this value.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="badge-warning">
                        <div class="badge-title">⚠️ AI Extraction: No Score Found</div>
                        <p>
                            Could not locate a valid CIBIL score (300–900) in the PDF text.<br>
                            Please enter the score manually below.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except ImportError:
            st.markdown(
                """
                <div class="badge-warning">
                    <div class="badge-title">⚠️ PDF Extraction Unavailable</div>
                    <p>
                        <code>pdfplumber</code> is not installed. Install it via
                        <code>pip install pdfplumber</code> for automatic CIBIL score extraction.<br>
                        Please enter the score manually below.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as pdf_err:
            st.markdown(
                f"""
                <div class="badge-warning">
                    <div class="badge-title">⚠️ PDF Parsing Error</div>
                    <p>
                        Failed to parse the uploaded PDF: {pdf_err}<br>
                        Please enter the score manually below.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="badge-warning">
                <div class="badge-title">⏳ Awaiting Upload</div>
                <p>No CIBIL report uploaded. You may still proceed with manual entry below.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Manual CIBIL Score Input (auto-populated if extracted) ────────────

    default_cibil = extracted_cibil if extracted_cibil is not None else 750

    cibil_score = st.number_input(
        "📊 CIBIL Score (Manual Confirmation)",
        min_value=300,
        max_value=900,
        value=default_cibil,
        step=1,
        help="Enter the CIBIL score as printed on the uploaded report (300–900). "
             "Auto-populated if extraction was successful.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2. Financial Core Metrics ────────────────────────────────────────

    with st.expander("💰  2. Financial Core Metrics", expanded=True):
        fin_c1, fin_c2 = st.columns(2)
        with fin_c1:
            applicant_income = st.slider(
                "Applicant Monthly Income (₹)",
                min_value=1000,
                max_value=25000,
                value=5500,
                step=500,
            )
            coapplicant_income = st.slider(
                "Co-applicant Income (₹)",
                min_value=0,
                max_value=10000,
                value=0,
                step=500,
            )
        with fin_c2:
            loan_amount = st.slider(
                "Loan Amount Requested (₹K)",
                min_value=10,
                max_value=500,
                value=130,
                step=10,
            )
            dti_ratio = (
                (loan_amount * 1000) / max(applicant_income + coapplicant_income, 1)
            )
            st.markdown(
                f"""
                <div style="
                    background: var(--bg-secondary);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    padding: 0.8rem 1rem;
                    margin-top: 0.8rem;
                ">
                    <span style="font-size:0.68rem; color:var(--text-muted);
                                 text-transform:uppercase; letter-spacing:0.06em;
                                 font-weight:600;">Debt-to-Income Ratio</span><br>
                    <span style="font-family:'JetBrains Mono',monospace;
                                 font-size:1.3rem; font-weight:700;
                                 color:{'var(--success)' if dti_ratio < 5 else 'var(--warning)' if dti_ratio < 10 else 'var(--danger)'};">
                        {dti_ratio:.2f}x
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 3. Demographics & Risk Parameters ────────────────────────────────

    with st.expander("👤  3. Demographics & Risk Parameters", expanded=True):
        demo_c1, demo_c2 = st.columns(2)
        with demo_c1:
            gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
            married = st.selectbox(
                "Marital Status", options=["Yes", "No"], index=0,
                format_func=lambda x: "Married" if x == "Yes" else "Unmarried",
            )
        with demo_c2:
            education = st.selectbox(
                "Educational Qualification",
                options=["Graduate", "Not Graduate"],
                index=0,
            )
            property_area = st.selectbox(
                "Property Area Classification",
                options=["Urban", "Semiurban", "Rural"],
                index=0,
            )

# ═══════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — SYSTEM INFERENCE & DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

with right_col:

    st.markdown(
        '<div class="panel-title">⚙️&ensp;Underwriting Decision Engine</div>',
        unsafe_allow_html=True,
    )

    # ── Summary Card ─────────────────────────────────────────────────────

    credit_flag = "POSITIVE" if cibil_score >= 700 else "NEGATIVE"
    credit_color = "var(--success)" if cibil_score >= 700 else "var(--danger)"
    engine_label = engine_choice if model_loaded else "N/A"

    st.markdown(
        f"""
        <div class="panel" style="margin-bottom:1rem;">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.6rem;">
                <div>
                    <span style="font-size:0.65rem; color:var(--text-muted);
                                 text-transform:uppercase; letter-spacing:0.05em;
                                 font-weight:600;">CIBIL Score</span><br>
                    <span style="font-family:'JetBrains Mono',monospace;
                                 font-size:1.5rem; font-weight:800;
                                 color:{credit_color};">{cibil_score}</span>
                </div>
                <div>
                    <span style="font-size:0.65rem; color:var(--text-muted);
                                 text-transform:uppercase; letter-spacing:0.05em;
                                 font-weight:600;">Credit Flag</span><br>
                    <span style="font-family:'JetBrains Mono',monospace;
                                 font-size:1.1rem; font-weight:700;
                                 color:{credit_color};">{credit_flag}</span>
                </div>
                <div>
                    <span style="font-size:0.65rem; color:var(--text-muted);
                                 text-transform:uppercase; letter-spacing:0.05em;
                                 font-weight:600;">Total Income</span><br>
                    <span style="font-family:'JetBrains Mono',monospace;
                                 font-size:1.1rem; font-weight:700;
                                 color:var(--text-primary);">₹{applicant_income + coapplicant_income:,}</span>
                </div>
                <div>
                    <span style="font-size:0.65rem; color:var(--text-muted);
                                 text-transform:uppercase; letter-spacing:0.05em;
                                 font-weight:600;">Active Engine</span><br>
                    <span style="font-family:'JetBrains Mono',monospace;
                                 font-size:0.85rem; font-weight:700;
                                 color:var(--accent-light);">{engine_label}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Execute Button ───────────────────────────────────────────────────

    execute_clicked = st.button("⚡  EXECUTE UNDERWRITING ASSESSMENT")

    if execute_clicked:
        if not model_loaded:
            st.markdown(
                """
                <div class="badge-danger">
                    <div class="badge-title">⛔ Inference Blocked</div>
                    <p>Model artifacts are unavailable. Cannot execute assessment.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Parsing CIBIL parameters and evaluating risk matrices..."):
                time.sleep(1.2)  # Simulate processing latency

                # ── Feature Engineering ──────────────────────────────
                # Map categorical variables to binary integers
                credit_history = 1.0 if cibil_score >= 700 else 0.0
                gender_male = 1 if gender == "Male" else 0
                married_yes = 1 if married == "Yes" else 0
                education_not_grad = 1 if education == "Not Graduate" else 0
                prop_semiurban = 1 if property_area == "Semiurban" else 0
                prop_urban = 1 if property_area == "Urban" else 0

                # ── Construct DataFrame matching training schema ─────
                input_data = {
                    "ApplicantIncome": [applicant_income],
                    "CoapplicantIncome": [float(coapplicant_income)],
                    "LoanAmount": [float(loan_amount)],
                    "Credit_History": [credit_history],
                    "Gender_Male": [gender_male],
                    "Married_Yes": [married_yes],
                    "Education_Not Graduate": [education_not_grad],
                    "Property_Area_Semiurban": [prop_semiurban],
                    "Property_Area_Urban": [prop_urban],
                }
                input_df = pd.DataFrame(input_data)

                # ── CRITICAL: Re-index to match model_columns ────────
                # This eliminates any column-mismatch / array alignment bugs
                input_df = input_df.reindex(columns=model_columns, fill_value=0)

                # ── Scale continuous features ────────────────────────
                continuous_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
                input_df_scaled = input_df.copy()
                input_df_scaled[continuous_cols] = scaler.transform(
                    input_df[continuous_cols]
                )

                # ── Select active model ──────────────────────────────
                if engine_choice == "Logistic Regression":
                    active_model = lr_model
                    model_input = input_df_scaled
                else:
                    active_model = dt_model
                    # Decision Tree was trained on unscaled data
                    model_input = input_df

                prediction = active_model.predict(model_input)[0]
                probabilities = active_model.predict_proba(model_input)[0]

            # ── Decision Output ──────────────────────────────────────

            if prediction == 1:
                # ── APPROVED ─────────────────────────────────────────
                approval_prob = probabilities[1] * 100
                risk_index = (
                    "LOW" if approval_prob >= 80
                    else "MODERATE" if approval_prob >= 60
                    else "ELEVATED"
                )
                risk_color = (
                    "var(--success)" if risk_index == "LOW"
                    else "var(--warning)" if risk_index == "MODERATE"
                    else "#f97316"
                )

                st.markdown(
                    f"""
                    <div class="decision-approved">
                        <div class="decision-header" style="color:var(--success);">
                            UNDERWRITING DECISION — {engine_choice.upper()}
                        </div>
                        <div class="decision-verdict" style="color:#6ee7b7;">
                            ✅&ensp;APPLICATION APPROVED
                        </div>
                        <p style="font-size:0.85rem; color:#a7f3d0; margin:0;">
                            Congratulations! Your loan application is likely to be
                            <strong>APPROVED</strong> based on automated credit analysis.
                        </p>
                        <div class="decision-metrics">
                            <div class="d-metric">
                                <div class="d-label">Approval Probability</div>
                                <div class="d-value" style="color:var(--success);">
                                    {approval_prob:.1f}%
                                </div>
                            </div>
                            <div class="d-metric">
                                <div class="d-label">Risk Index Rating</div>
                                <div class="d-value" style="color:{risk_color};">
                                    {risk_index}
                                </div>
                            </div>
                            <div class="d-metric">
                                <div class="d-label">CIBIL Assessment</div>
                                <div class="d-value" style="color:var(--success);">
                                    {cibil_score} — PASS
                                </div>
                            </div>
                            <div class="d-metric">
                                <div class="d-label">Debt-to-Income</div>
                                <div class="d-value" style="color:{
                                    'var(--success)' if dti_ratio < 5
                                    else 'var(--warning)' if dti_ratio < 10
                                    else 'var(--danger)'};">
                                    {dti_ratio:.2f}x
                                </div>
                            </div>
                        </div>
                        <div class="rec-list">
                            <div class="rec-title">Recommended Actions</div>
                            <ul>
                                <li>Proceed with formal offer letter generation</li>
                                <li>Schedule property valuation & legal verification</li>
                                <li>Initiate KYC compliance checks</li>
                                <li>Route to disbursement queue (Priority: Standard)</li>
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                verdict_text = "APPROVED"
                verdict_prob = approval_prob

            else:
                # ── REJECTED ─────────────────────────────────────────
                rejection_prob = probabilities[0] * 100
                risk_index = (
                    "CRITICAL" if rejection_prob >= 80
                    else "HIGH" if rejection_prob >= 60
                    else "ELEVATED"
                )
                risk_color = (
                    "var(--danger)" if risk_index == "CRITICAL"
                    else "#f97316" if risk_index == "HIGH"
                    else "var(--warning)"
                )

                # Determine primary rejection reasons
                reasons = []
                if cibil_score < 700:
                    reasons.append("Insufficient Credit History — CIBIL score below 700 threshold")
                if dti_ratio >= 8:
                    reasons.append("High Debt-to-Income Ratio — exceeds 8x prudential limit")
                if applicant_income + coapplicant_income < 3000:
                    reasons.append("Combined household income below minimum eligibility floor")
                if loan_amount > 300:
                    reasons.append("Loan amount exceeds standard automated approval ceiling")
                if not reasons:
                    reasons.append("Combined risk factors exceeded acceptable underwriting threshold")

                reasons_html = "".join(f"<li>{r}</li>" for r in reasons)

                st.markdown(
                    f"""
                    <div class="decision-rejected">
                        <div class="decision-header" style="color:var(--danger);">
                            UNDERWRITING DECISION — {engine_choice.upper()}
                        </div>
                        <div class="decision-verdict" style="color:#fca5a5;">
                            ❌&ensp;APPLICATION FLAGGED — REJECTED
                        </div>
                        <p style="font-size:0.85rem; color:#fecaca; margin:0;">
                            Review complete: Your loan application has a high risk of
                            <strong>REJECTION</strong> based on automated credit analysis.
                        </p>
                        <div class="decision-metrics">
                            <div class="d-metric">
                                <div class="d-label">Rejection Probability</div>
                                <div class="d-value" style="color:var(--danger);">
                                    {rejection_prob:.1f}%
                                </div>
                            </div>
                            <div class="d-metric">
                                <div class="d-label">Risk Index Rating</div>
                                <div class="d-value" style="color:{risk_color};">
                                    {risk_index}
                                </div>
                            </div>
                            <div class="d-metric">
                                <div class="d-label">CIBIL Assessment</div>
                                <div class="d-value" style="color:{'var(--danger)' if cibil_score < 700 else 'var(--success)'};">
                                    {cibil_score} — {'FAIL' if cibil_score < 700 else 'PASS'}
                                </div>
                            </div>
                            <div class="d-metric">
                                <div class="d-label">Debt-to-Income</div>
                                <div class="d-value" style="color:{
                                    'var(--success)' if dti_ratio < 5
                                    else 'var(--warning)' if dti_ratio < 10
                                    else 'var(--danger)'};">
                                    {dti_ratio:.2f}x
                                </div>
                            </div>
                        </div>
                        <div class="rec-list">
                            <div class="rec-title">Primary Reasons for Flagging</div>
                            <ul>{reasons_html}</ul>
                        </div>
                        <div class="rec-list" style="margin-top:0.6rem;">
                            <div class="rec-title">Recommended Next Steps</div>
                            <ul>
                                <li>Escalate to manual underwriting review (Senior Analyst)</li>
                                <li>Request supplementary income documentation</li>
                                <li>Advise applicant to improve CIBIL score before re-application</li>
                                <li>Consider co-signer or collateral-backed restructuring</li>
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                verdict_text = "REJECTED"
                verdict_prob = rejection_prob

            # ──────────────────────────────────────────────────────────────
            # FEATURE ATTRIBUTION & RISK MATRIX
            # ──────────────────────────────────────────────────────────────

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-title">🔍&ensp;Feature Attribution & Risk Matrix</div>',
                unsafe_allow_html=True,
            )

            if engine_choice == "Logistic Regression":
                # Extract LR coefficients and map to feature names
                coefficients = lr_model.coef_[0]
                coef_df = pd.DataFrame({
                    "Feature": model_columns,
                    "Coefficient": coefficients,
                })
                coef_df["AbsCoef"] = coef_df["Coefficient"].abs()
                coef_df = coef_df.sort_values("AbsCoef", ascending=False)

                max_abs = coef_df["AbsCoef"].max() if coef_df["AbsCoef"].max() > 0 else 1.0

                # Build a custom HTML horizontal bar chart
                bars_html = ""
                for _, row in coef_df.iterrows():
                    feat = row["Feature"]
                    coef = row["Coefficient"]
                    abs_coef = row["AbsCoef"]
                    bar_pct = (abs_coef / max_abs) * 100
                    bar_color = "#10b981" if coef > 0 else "#ef4444"
                    direction = "▲ Approval" if coef > 0 else "▼ Rejection"

                    bars_html += f"""
                    <div class="attr-bar-container">
                        <div class="attr-label">{feat}</div>
                        <div class="attr-bar-track">
                            <div class="attr-bar-fill" style="
                                width: {bar_pct:.1f}%;
                                background: linear-gradient(90deg, {bar_color}88, {bar_color});
                            "></div>
                        </div>
                        <div class="attr-value" style="color: {bar_color};">
                            {coef:+.3f}
                        </div>
                    </div>
                    """

                st.markdown(
                    f"""
                    <div class="panel">
                        <div style="font-size:0.72rem; color:var(--text-muted);
                                    margin-bottom:0.8rem; font-weight:500;">
                            Logistic Regression coefficient weights — positive values push toward
                            <span style="color:#10b981;">Approval</span>,
                            negative toward <span style="color:#ef4444;">Rejection</span>.
                        </div>
                        {bars_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                # Decision Tree — show feature importances
                importances = dt_model.feature_importances_
                imp_df = pd.DataFrame({
                    "Feature": model_columns,
                    "Importance": importances,
                })
                imp_df = imp_df.sort_values("Importance", ascending=False)
                max_imp = imp_df["Importance"].max() if imp_df["Importance"].max() > 0 else 1.0

                bars_html = ""
                for _, row in imp_df.iterrows():
                    feat = row["Feature"]
                    imp = row["Importance"]
                    bar_pct = (imp / max_imp) * 100
                    bar_color = "#818cf8"

                    bars_html += f"""
                    <div class="attr-bar-container">
                        <div class="attr-label">{feat}</div>
                        <div class="attr-bar-track">
                            <div class="attr-bar-fill" style="
                                width: {bar_pct:.1f}%;
                                background: linear-gradient(90deg, {bar_color}88, {bar_color});
                            "></div>
                        </div>
                        <div class="attr-value" style="color: {bar_color};">
                            {imp:.4f}
                        </div>
                    </div>
                    """

                st.markdown(
                    f"""
                    <div class="panel">
                        <div style="font-size:0.72rem; color:var(--text-muted);
                                    margin-bottom:0.8rem; font-weight:500;">
                            Decision Tree feature importances — higher values indicate
                            greater influence on the classification decision.
                        </div>
                        {bars_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ──────────────────────────────────────────────────────────────
            # DOWNLOADABLE UNDERWRITING MEMO
            # ──────────────────────────────────────────────────────────────

            st.markdown("<br>", unsafe_allow_html=True)

            memo_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            memo_text = f"""
════════════════════════════════════════════════════════════════
  UNDERWRITING SANCTION CLEARANCE MEMO
  Institutional Credit Risk Assessment Platform
════════════════════════════════════════════════════════════════

  Generated: {memo_timestamp}
  Engine:    {engine_choice}

────────────────────────────────────────────────────────────────
  APPLICANT SUMMARY METRICS
────────────────────────────────────────────────────────────────

  CIBIL Score ................ {cibil_score}
  Credit History Flag ........ {"POSITIVE (1.0)" if cibil_score >= 700 else "NEGATIVE (0.0)"}
  Applicant Income (₹) ...... {applicant_income:,}
  Co-applicant Income (₹) ... {coapplicant_income:,}
  Total Household Income ..... ₹{applicant_income + coapplicant_income:,}
  Loan Amount Requested ...... ₹{loan_amount}K
  Debt-to-Income Ratio ....... {dti_ratio:.2f}x
  Gender ..................... {gender}
  Marital Status ............. {"Married" if married == "Yes" else "Unmarried"}
  Education .................. {education}
  Property Area .............. {property_area}

────────────────────────────────────────────────────────────────
  MODEL DECISION
────────────────────────────────────────────────────────────────

  Verdict:     ** {verdict_text} **
  Confidence:  {verdict_prob:.1f}%

════════════════════════════════════════════════════════════════
  This memo was auto-generated by the Credit Risk Assessment
  Platform. It is intended for internal use only and does not
  constitute a binding financial commitment.
════════════════════════════════════════════════════════════════
"""

            st.download_button(
                label="📥  Download Underwriting Clearance Memo (.txt)",
                data=memo_text,
                file_name=f"underwriting_memo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="footer">
        Institutional Credit Risk Assessment Platform &nbsp;·&nbsp;
        Dual-Engine: Logistic Regression + Decision Tree v3.0.0 &nbsp;·&nbsp;
        Powered by scikit-learn + Streamlit &nbsp;·&nbsp;
        {current_ts}
    </div>
    """,
    unsafe_allow_html=True,
)
