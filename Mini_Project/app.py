# =========================
# STREAMLIT APP
# =========================

import html

import streamlit as st
from common import generate_summary, load_model_and_tokenizer


st.set_page_config(
    page_title="Summarize AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f7faff;
            --panel: rgba(255, 255, 255, 0.88);
            --panel-strong: rgba(255, 255, 255, 0.98);
            --text: #10213a;
            --muted: #5b6b84;
            --accent: #0f9d8d;
            --accent-2: #1d4ed8;
            --accent-soft: rgba(15, 157, 141, 0.12);
            --border: rgba(15, 33, 58, 0.08);
            --shadow: 0 20px 60px rgba(22, 42, 72, 0.12);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(29, 78, 216, 0.10), transparent 24%),
                radial-gradient(circle at top right, rgba(15, 157, 141, 0.12), transparent 24%),
                linear-gradient(180deg, #f8fbff 0%, #f4f8fe 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4, p, label, div {
            color: var(--text);
        }

        .hero {
            padding: 1.8rem 2rem;
            border: 1px solid var(--border);
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(244,248,255,0.94));
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.9rem;
        }

        .hero-title {
            font-size: clamp(2.2rem, 4vw, 4.1rem);
            line-height: 1.02;
            font-weight: 800;
            margin: 0;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            line-height: 1.7;
            color: var(--muted);
            max-width: 60ch;
            margin-top: 0.9rem;
        }

        .surface {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 1.25rem 1.25rem 1rem;
            backdrop-filter: blur(18px);
        }

        .surface-strong {
            background: var(--panel-strong);
        }

        .metric-card {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(247,250,255,0.96));
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .metric-value {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text);
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .summary-box {
            background: linear-gradient(135deg, rgba(15, 157, 141, 0.10), rgba(29, 78, 216, 0.08));
            border: 1px solid rgba(15, 157, 141, 0.14);
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            box-shadow: var(--shadow);
        }

        .sample-chip {
            display: inline-block;
            margin: 0.25rem 0.35rem 0 0;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            background: rgba(29, 78, 216, 0.08);
            color: #1d4ed8;
            border: 1px solid rgba(29, 78, 216, 0.12);
            font-size: 0.88rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%);
            border-right: 1px solid rgba(15, 33, 58, 0.08);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        .stTextArea textarea {
            border-radius: 20px !important;
            border: 1px solid rgba(15, 33, 58, 0.12) !important;
            background: rgba(255, 255, 255, 0.98) !important;
            color: var(--text) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
        }

        .stButton button {
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: white;
            font-weight: 700;
            padding: 0.75rem 1.2rem;
            box-shadow: 0 14px 30px rgba(29, 78, 216, 0.24);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 34px rgba(29, 78, 216, 0.28);
        }

        .stAlert {
            border-radius: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_summarizer_components():
    return load_model_and_tokenizer()


tokenizer, model, device = get_summarizer_components()
device_label = "GPU" if str(device).startswith("cuda") else "CPU"

with st.sidebar:
    st.markdown("## ✨ Summarize AI")
    st.caption("A bright, professional interface for abstractive summarization.")
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Active device</span>
            <span class="metric-value">{device_label}</span>
            <span class="metric-note">Model is loaded once and cached for fast inference.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Try a sample")
    samples = [
        "A startup unveiled a solar-powered delivery drone designed to cut urban traffic.",
        "Researchers reported a new battery breakthrough that could improve EV charging speed.",
        "The city council approved a waterfront redevelopment plan focused on green spaces.",
    ]
    for sample in samples:
        st.markdown(f"<span class='sample-chip'>{sample}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Notes")
    st.write("• Paste a news article or long-form text.")
    st.write("• Click Summarize to generate a concise abstractive summary.")
    st.write("• GPU is used automatically when available.")

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Abstractive Summarization · BART · GPU-ready</div>
        <h1 class="hero-title">Turn long articles into polished, readable summaries.</h1>
        <p class="hero-subtitle">
            Paste text, press one button, and get a concise summary powered by a fine-tuned BART model.
            The interface is built to feel calm, modern, and production-grade.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.markdown("### Your article")
    text = st.text_area(
        "Enter your article",
        height=320,
        placeholder="Paste a news story, blog post, report, or any long-form text here...",
        label_visibility="collapsed",
    )
    summarize_clicked = st.button("Summarize now", use_container_width=True)

with right:
    st.markdown("### Model snapshot")
    snapshot_cols = st.columns(2)
    with snapshot_cols[0]:
        st.markdown(
            """
            <div class="metric-card">
                <span class="metric-label">Model</span>
                <span class="metric-value">BART</span>
                <span class="metric-note">facebook/bart-large-cnn</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with snapshot_cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="metric-label">Runtime</span>
                <span class="metric-value">{device_label}</span>
                <span class="metric-note">Auto-detected at launch</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### How it works")
    st.write("1. The text is tokenized and moved to the active device.")
    st.write("2. The model generates a compact abstractive summary.")
    st.write("3. The result is shown below in a highlighted panel.")

if summarize_clicked:
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        summary = generate_summary(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        st.markdown("### Summary")
        st.markdown(
            f"""
            <div class="summary-box">
                {html.escape(summary)}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div style="margin-top: 1.5rem; text-align: center; color: #66758f; font-size: 0.92rem;">
        Built for fast inference, bright visuals, and a cleaner summarization workflow.
    </div>
    """,
    unsafe_allow_html=True,
)