import sys
import logging
import warnings
import streamlit as st
from common import generate_summary, load_model_and_tokenizer
from transformers import logging as transformers_logging

# --- LOGGING (idempotent to avoid duplicate handlers on Streamlit reruns) ---
logger = logging.getLogger("summarize_ai")
if not logger.handlers:
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)

# Reduce verbosity of noisy libs
for noisy in ("datasets", "torch", "urllib3", "PIL", "tokenizers", "nltk"):  # common noisy loggers
    logging.getLogger(noisy).setLevel(logging.ERROR)

# Configure transformers logger and suppress specific warnings about __path__ aliasing
transformers_logging.set_verbosity_error()
try:
    transformers_logging.disable_default_handler()
except Exception:
    pass
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*Accessing `__path__`.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*was truncated to the first.*")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Summarize AI",
    page_icon="✨",
    layout="centered",
)

# --- STYLING ---
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0b2321 0%, #071717 35%, #041212 100%);
        color: #e6f6f3;
    }
    .header {
        padding: 18px 0;
        border-radius: 8px;
        background: linear-gradient(90deg, #0b6b63, #063d36);
        color: #e8fff9;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .stTextArea textarea {
        border-radius: 10px;
        background-color: #0a1817 !important;
        color: #e6f6f3 !important;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #0a1817 !important;
        color: #e6f6f3 !important;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 3.1em;
        background-color: #0f9d8d;
        color: #041212;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .summary-container {
        padding: 18px;
        background-color: #071717;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.6);
        border-left: 6px solid #0f9d8d;
        color: #dffaf4;
        line-height: 1.6;
    }
    .meta { color: #bfeee6; font-size: 0.95em; }
    .css-1v0mbdj.egzxvld2 { background-color: transparent; } /* container spacing fix */
    </style>
    """,
    unsafe_allow_html=True,
)


# --- DATA & MODEL ---
@st.cache_resource
def get_summarizer_components():
    return load_model_and_tokenizer()

with st.spinner("Loading model and tokenizer..."):
    tokenizer, model, device = get_summarizer_components()
    logger.info("Model and tokenizer loaded; device=%s", device)

device_label = "GPU (CUDA)" if "cuda" in str(device).lower() else "CPU"


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style='text-align:center'>
      <h2>✨ Summarize AI</h2>
      <div class='meta'>Running on: <strong>""" + device_label + """</strong></div>
    </div>
    
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Generation Controls")
    st.caption("Tweak these to control summary length and quality. Hover each control for a short hint.")

    st.divider()
    st.subheader("Summary size")
    summary_style_label = st.selectbox(
        "Summary style",
        ["Balanced", "Crisp", "Ultra-Short", "Bullet Input Mode"],
        index=0,
        help="Balanced = general use, Crisp = tighter wording, Ultra-Short = maximum compression, Bullet Input Mode = better for task lists.",
    )

    style_map = {
        "Balanced": "balanced",
        "Crisp": "crisp",
        "Ultra-Short": "ultra_short",
        "Bullet Input Mode": "bullet_mode",
    }
    summary_style = style_map[summary_style_label]

    preset = st.selectbox("Preset", ["Short", "Medium", "Long"], index=1, help="Quick presets: Short = concise, Medium = balanced, Long = more detail")
    if preset == "Short":
        max_summary_default = 60
        min_summary_default = 18
    elif preset == "Long":
        max_summary_default = 220
        min_summary_default = 60
    else:
        max_summary_default = 130
        min_summary_default = 45

    max_summary_length = st.slider(
        "Max summary length",
        20,
        400,
        max_summary_default,
        help="Approximate maximum tokens (words) for the summary — increase for longer summaries.",
    )
    min_summary_length = st.slider(
        "Min summary length",
        10,
        200,
        min_summary_default,
        help="Minimum tokens (words) for the summary — prevents extremely short outputs.",
    )
    num_beams = st.slider(
        "Beams (quality vs speed)",
        1,
        8,
        4,
        help="Higher beams improve quality but increase generation time; 1 is fastest.",
    )
    candidate_count = st.slider(
        "Candidates (reranking)",
        1,
        5,
        3,
        help="Generate multiple candidates and pick the best. Higher values can improve quality but are slower.",
    )

    st.divider()
    st.caption("Powered by your local BART model")


# --- MAIN UI ---
st.markdown("<div class='header'><h1 style='margin:0'>Summarize AI</h1></div>", unsafe_allow_html=True)
st.markdown("Convert long articles into concise, readable summaries — fast.")

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

col_left, col_right = st.columns([3, 1])

with col_left:
    uploaded_file = st.file_uploader("Upload a .txt file (optional)")
    if uploaded_file is not None:
        try:
            raw = uploaded_file.read().decode('utf-8')
            st.session_state.input_text = raw
        except Exception:
            st.error("Could not read uploaded file. Please upload a UTF-8 encoded text file.")

    input_text = st.text_area(
        "Paste your text here:",
        value=st.session_state.input_text,
        height=300,
        placeholder="e.g., The latest report on climate change suggests...",
    )

with col_right:
    st.subheader("Options")
    st.caption("Adjust summary length and quality")
    st.markdown("\n")
    st.caption(f"Original length: {len(st.session_state.input_text.split())} words")

generate = st.button("Generate Summary")

if generate:
    if input_text.strip():
        with st.spinner("Generating summary — this may take a moment..."):
            try:
                logger.info("Generating summary: input_words=%d, max=%d, min=%d, beams=%d", len(input_text.split()), max_summary_length, min_summary_length, num_beams)
                summary = generate_summary(
                    text=input_text,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_summary_length=max_summary_length,
                    min_summary_length=min_summary_length,
                    num_beams=num_beams,
                    candidate_count=candidate_count,
                    summary_style=summary_style,
                )
                logger.info("Summary generation complete: summary_words=%d", len(summary.split()))
                st.subheader("Result")
                st.markdown(f'<div class="summary-container">{summary}</div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                col1.caption(f"Original: {len(input_text.split())} words")
                col2.caption(f"Summary: {len(summary.split())} words")

                with st.expander("Show original text"):
                    st.write(input_text)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter or upload some text first.")

st.divider()
st.caption("Built for fast, abstractive summarization.")